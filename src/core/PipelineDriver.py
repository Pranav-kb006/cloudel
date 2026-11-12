"""Pipeline driver: overlap layer downloads with deterministic synthetic compute.

``PipelineDriver`` consumes layers in order from an :class:`AsyncPrefetcher`,
runs a real (but intentionally small) FP32 matvec/elementwise kernel on the
downloaded bytes, evicts the layer after use, and records timings plus RSS.

The compute engine is a **stand-in** for transformer GEMMs.  It is real
arithmetic over the downloaded bytes, not a sleep, and its cost is tunable with
``work_factor``.  It is not a transformer and must not be presented as one.

TTFT definition
---------------
A real autoregressive first token requires the **full forward pass** (every
layer), so ``ttft_s`` is measured from the start of ``run()`` until the last
layer's compute completes.  ``first_layer_s`` is also reported (completion of
layer 0's compute) as the streaming-fill latency; it is **not** an LLM TTFT.
``first_payload_s`` records arrival of the first layer's bytes.

Masking ratio
-------------
``masking_ratio = compute_s / (compute_s + io_wait_s)``.  It is the fraction of
consumer-side time spent computing rather than waiting for bytes, in [0, 1].
``io_masking_ratio = 1 - io_wait_s / total_s`` additionally accounts for
startup/teardown; both are reported.

STATUS:
  Built: per-layer consume/compute/evict loop, deterministic synthetic FP32
         (plus F16/BF16/int) matvec + elementwise kernel with configurable
         work_factor, TTFT/I-O-masking/RSS metrics, prefetcher pacing feedback.
  TBD:   Real transformer GEMMs require torch/llama.cpp (not installed); layer
         spans containing mixed dtypes are decoded as the layer's shared dtype
         when known, else F32.  Full-model parameter preallocation is not
         claimed as a memory win.
  Packages: Python standard library only.  psutil is optional (RSS fallback is
            /proc/self/statm); aiohttp is optional; torch/numpy are NOT required.
"""

from __future__ import annotations

import array
import asyncio
import hashlib
import math
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .AsyncPrefetcher import AsyncPrefetcher, LayerPayload
from .MemoryBuffer import MemoryBuffer, get_rss_mb

__all__ = [
    "PipelineDriver",
    "SyntheticComputeEngine",
    "ComputeResult",
    "decode_tensor_bytes",
]

_STRUCT_CODES: dict[str, str] = {
    "F64": "d",
    "F32": "f",
    "F16": "e",
    "I64": "q",
    "I32": "i",
    "I16": "h",
    "I8": "b",
    "U64": "Q",
    "U32": "I",
    "U16": "H",
    "U8": "B",
    "BOOL": "?",
}
_ITEMSIZE: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def decode_tensor_bytes(data: bytes, dtype: str = "F32") -> Sequence[float]:
    """Decode a tensor byte string into a sequence of numbers.

    Supports the common safetensors numeric dtypes using the stdlib ``array``
    module plus a manual BF16 decoder.  Unsupported dtypes raise ``ValueError``;
    the compute engine catches that and falls back to byte-wise arithmetic.
    """
    dtype = dtype.upper()
    if dtype == "BF16":
        usable = len(data) // 2 * 2
        values = [0.0] * (usable // 2)
        for index in range(usable // 2):
            bits = int.from_bytes(data[2 * index : 2 * index + 2], "little")
            values[index] = struct.unpack("<f", (bits << 16).to_bytes(4, "little"))[0]
        return values
    code = _STRUCT_CODES.get(dtype)
    if code is None:
        raise ValueError(f"unsupported dtype for synthetic compute: {dtype!r}")
    itemsize = _ITEMSIZE[dtype]
    usable = len(data) // itemsize * itemsize
    decoded = array.array(code)
    decoded.frombytes(data[:usable])
    if sys.byteorder == "big" and itemsize > 1:
        decoded.byteswap()
    return decoded


@dataclass(slots=True)
class ComputeResult:
    """Outcome of one synthetic layer kernel."""

    layer: str
    checksum: float
    dtype: str
    tensor_elements: int
    processed_elements: int
    operations: int
    repeats: int
    seconds: float
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "checksum": self.checksum,
            "dtype": self.dtype,
            "tensor_elements": self.tensor_elements,
            "processed_elements": self.processed_elements,
            "operations": self.operations,
            "repeats": self.repeats,
            "seconds": self.seconds,
            "fallback": self.fallback,
        }


class SyntheticComputeEngine:
    """Deterministic, tunable stand-in for a transformer layer forward pass.

    The kernel decodes the layer bytes, then repeats a dense ``d x d`` matvec
    plus an elementwise reduction.  ``work_factor`` scales the number of
    repetitions (values below 1 shrink the processed prefix).  Results depend
    only on the layer name, bytes, seed and work factor, so repeated runs are
    comparable.

    Parameters
    ----------
    work_factor:
        Compute intensity multiplier.  ``1.0`` is one full pass.
    hidden_dim:
        Matvec dimension; clamped to ``isqrt(processed_elements)``.
    max_elements:
        Cap on elements processed per layer (default 262144).  This keeps a
        large real checkpoint from turning the CPU benchmark into a multi-minute
        run; the prefix that is processed is genuine downloaded data.
    seed:
        Seed mixed into the deterministic coefficient vector.
    dtype:
        Fallback dtype when the payload does not carry one (default F32).
    """

    def __init__(
        self,
        work_factor: float = 1.0,
        hidden_dim: int = 32,
        *,
        max_elements: int = 262_144,
        seed: int = 0,
        dtype: str = "F32",
    ) -> None:
        if work_factor < 0:
            raise ValueError("work_factor must be >= 0")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if max_elements < 1:
            raise ValueError("max_elements must be >= 1")
        self.work_factor = float(work_factor)
        self.hidden_dim = int(hidden_dim)
        self.max_elements = int(max_elements)
        self.seed = int(seed)
        self.dtype = dtype.upper()

    def _coefficients(self, layer_name: str, count: int) -> list[float]:
        digest = hashlib.blake2b(
            layer_name.encode("utf-8"), digest_size=8, key=str(self.seed).encode()
        ).digest()
        state = int.from_bytes(digest, "little") or 0x9E3779B97F4A7C15
        mask = (1 << 64) - 1
        values: list[float] = []
        for _ in range(count):
            state = (state * 6364136223846793005 + 1442695040888963407) & mask
            values.append((state >> 11) / float(1 << 53) - 0.5)
        return values

    def run_layer(
        self,
        layer_name: str,
        data: bytes,
        *,
        dtype: str | None = None,
        work_factor: float | None = None,
    ) -> ComputeResult:
        started = time.perf_counter()
        chosen_dtype = (dtype or self.dtype).upper()
        fallback = False
        try:
            values = decode_tensor_bytes(data, chosen_dtype)
            tensor_elements = len(values)
        except (ValueError, struct.error):
            # Unknown dtype: still do real arithmetic over every byte.
            values = list(data)
            tensor_elements = len(values)
            chosen_dtype = "U8-fallback"
            fallback = True

        if tensor_elements == 0:
            return ComputeResult(
                layer=layer_name,
                checksum=0.0,
                dtype=chosen_dtype,
                tensor_elements=0,
                processed_elements=0,
                operations=0,
                repeats=1,
                seconds=time.perf_counter() - started,
                fallback=fallback,
            )

        wf = self.work_factor if work_factor is None else float(work_factor)
        if wf < 0:
            raise ValueError("work_factor must be >= 0")
        limit = min(tensor_elements, self.max_elements)
        if wf < 1.0:
            limit = max(1, int(limit * wf))
            repeats = 1
        else:
            repeats = max(1, int(round(wf)))
        sample = values[:limit]
        processed = len(sample)

        dimension = max(1, min(self.hidden_dim, math.isqrt(processed)))
        coefficients = self._coefficients(layer_name, dimension)
        accumulator = 0.0
        operations = 0

        for repetition in range(repeats):
            # Dense d x d matvec over a prefix of the downloaded tensor.
            for row in range(dimension):
                base = row * dimension
                total = 0.0
                for column in range(dimension):
                    total += float(sample[base + column]) * coefficients[column]
                accumulator += total * total
            operations += dimension * dimension
            # Elementwise reduction over the whole processed prefix.
            for index, value in enumerate(sample):
                accumulator += float(value) * coefficients[(index + repetition) % dimension]
            operations += processed
            accumulator = math.sqrt(abs(accumulator)) + 0.5 * accumulator

        checksum = accumulator / float(repeats)
        return ComputeResult(
            layer=layer_name,
            checksum=checksum,
            dtype=chosen_dtype,
            tensor_elements=tensor_elements,
            processed_elements=processed,
            operations=operations,
            repeats=repeats,
            seconds=time.perf_counter() - started,
            fallback=fallback,
        )


class PipelineDriver:
    """Consume prefetched layers, compute, evict, and return metrics.

    Parameters
    ----------
    prefetcher:
        An :class:`AsyncPrefetcher` (or compatible object exposing
        ``get``/``get_payload``, ``layer_names`` and optionally
        ``record_compute_time``/``buffer``).
    compute_engine:
        Object with ``run_layer(name, data, dtype=..., work_factor=...)``.
        Defaults to :class:`SyntheticComputeEngine`.
    buffer:
        Buffer to evict from after each layer; defaults to the prefetcher's
        ``buffer`` attribute.
    work_factor:
        Overrides the compute engine's work factor when provided.
    sample_rss:
        Sample RSS after each layer.
    compute_in_thread:
        Run the (CPU-bound) synthetic kernel via ``asyncio.to_thread`` so the
        prefetcher can keep scheduling downloads while layer *i* computes.
        Disable only for a custom engine that is not thread-safe.
    """

    def __init__(
        self,
        prefetcher: AsyncPrefetcher,
        compute_engine: Any = None,
        buffer: MemoryBuffer | None = None,
        *,
        work_factor: float | None = None,
        sample_rss: bool = True,
        compute_in_thread: bool = True,
    ) -> None:
        self.prefetcher = prefetcher
        self.compute_engine = (
            compute_engine
            if compute_engine is not None
            else SyntheticComputeEngine(
                work_factor=1.0 if work_factor is None else work_factor
            )
        )
        if work_factor is not None and hasattr(self.compute_engine, "work_factor"):
            self.compute_engine.work_factor = float(work_factor)
        self.work_factor = work_factor
        self.buffer = (
            buffer if buffer is not None else getattr(prefetcher, "buffer", None)
        )
        self.sample_rss = bool(sample_rss)
        self.compute_in_thread = bool(compute_in_thread)

    def _run_compute(self, name: str, data: bytes, dtype: str | None) -> Any:
        engine = self.compute_engine
        try:
            return engine.run_layer(
                name, data, dtype=dtype, work_factor=self.work_factor
            )
        except TypeError:
            # Tolerate a simpler custom engine with run_layer(name, data).
            return engine.run_layer(name, data)

    async def run(self, layer_names: Iterable[str] | None = None) -> dict[str, Any]:
        """Stream every requested layer and return a metrics dictionary."""
        started_here = False
        if not getattr(self.prefetcher, "started", False):
            await self.prefetcher.start()
            started_here = True

        names = (
            list(layer_names)
            if layer_names is not None
            else list(getattr(self.prefetcher, "layer_names", []))
        )
        if not names:
            raise ValueError("no layer names to run")

        t_start = time.perf_counter()
        rss_trace: list[dict[str, Any]] = []
        per_layer: list[dict[str, Any]] = []
        if self.sample_rss:
            rss_trace.append(
                {
                    "layer": None,
                    "phase": "start",
                    "rss_mb": get_rss_mb(),
                    "t": 0.0,
                }
            )

        ttft_s: float | None = None
        first_layer_s: float | None = None
        first_payload_s: float | None = None
        io_wait_total = 0.0
        compute_total = 0.0
        bytes_total = 0

        try:
            for name in names:
                request_started = time.perf_counter()
                payload = await self._get_payload(name)
                request_finished = time.perf_counter()
                io_wait = request_finished - request_started

                data = payload.data if isinstance(payload, LayerPayload) else bytes(payload)
                dtype = getattr(payload, "dtype", None)
                download_s = float(getattr(payload, "download_s", 0.0))
                retries = int(getattr(payload, "retries", 0))

                compute_started = time.perf_counter()
                if self.compute_in_thread:
                    result = await asyncio.to_thread(
                        self._run_compute, name, data, dtype
                    )
                else:
                    result = self._run_compute(name, data, dtype)
                compute_finished = time.perf_counter()
                compute_s = compute_finished - compute_started

                if self.buffer is not None:
                    self.buffer.evict(name)
                if hasattr(self.prefetcher, "record_compute_time"):
                    self.prefetcher.record_compute_time(compute_s)

                if first_payload_s is None:
                    first_payload_s = request_finished - t_start
                if first_layer_s is None:
                    first_layer_s = compute_finished - t_start

                io_wait_total += io_wait
                compute_total += compute_s
                bytes_total += len(data)
                rss_mb = get_rss_mb() if self.sample_rss else 0.0
                layer_metrics: dict[str, Any] = {
                    "layer": name,
                    "bytes": len(data),
                    "download_s": download_s,
                    "io_wait_s": io_wait,
                    "compute_s": compute_s,
                    "total_s": io_wait + compute_s,
                    "rss_mb": rss_mb,
                    "dtype": dtype,
                    "retries": retries,
                    "checksum": getattr(result, "checksum", result),
                    "elements": getattr(result, "processed_elements", None),
                    "operations": getattr(result, "operations", None),
                    "fallback": getattr(result, "fallback", False),
                }
                per_layer.append(layer_metrics)
                if self.sample_rss:
                    rss_trace.append(
                        {
                            "layer": name,
                            "phase": "after_compute",
                            "rss_mb": rss_mb,
                            "t": compute_finished - t_start,
                        }
                    )
            finished_at = time.perf_counter()
        finally:
            if started_here:
                await self.prefetcher.stop()

        total_s = finished_at - t_start
        ttft_s = total_s
        consumer_s = compute_total + io_wait_total
        masking_ratio = (
            compute_total / consumer_s if consumer_s > 0 else 1.0
        )
        io_masking_ratio = 1.0 - (io_wait_total / total_s) if total_s > 0 else 0.0

        metrics: dict[str, Any] = {
            "ttft_s": ttft_s if ttft_s is not None else 0.0,
            "first_layer_s": first_layer_s if first_layer_s is not None else 0.0,
            "first_payload_s": (
                first_payload_s if first_payload_s is not None else 0.0
            ),
            "total_s": total_s,
            "per_layer": per_layer,
            "io_wait_s": io_wait_total,
            "compute_s": compute_total,
            "masking_ratio": masking_ratio,
            "io_masking_ratio": max(0.0, min(1.0, io_masking_ratio)),
            "rss_trace": rss_trace,
            "bytes_downloaded": bytes_total,
            "layers": len(per_layer),
            "layer_names": list(names),
            "work_factor": self.work_factor,
            "compute_engine": type(self.compute_engine).__name__,
            "compute_in_thread": self.compute_in_thread,
        }
        if hasattr(self.prefetcher, "stats"):
            metrics["prefetcher_stats"] = self.prefetcher.stats()
        if self.buffer is not None:
            metrics["buffer_stats"] = self.buffer.stats()
        return metrics

    async def _get_payload(self, name: str) -> Any:
        getter = getattr(self.prefetcher, "get_payload", None)
        if getter is not None:
            return await getter(name)
        return await self.prefetcher.get(name)
