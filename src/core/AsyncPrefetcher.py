"""Asynchronous progressive layer prefetcher over HTTP Range requests.

The prefetcher keeps a bounded window of layers ahead of the consumer.  The
producer downloads layer payloads into a :class:`MemoryBuffer` (a synchronous
FIFO ring shared with the driver, not an ``asyncio.Queue``; it must support
eviction, pinning and RSS inspection, and all mutations stay on the loop
thread).  ``get()`` returns the bytes for one layer, waiting for the in-flight
fetch if necessary.  If the buffer dropped a prefetched payload because the
consumer was slower, the next ``get()`` simply re-fetches it.  There is
therefore no producer/consumer deadlock: progress always comes from an
on-demand fetch.

Pace heuristic
--------------
The base window is ``lookahead + 1`` layers (``lookahead=2`` -> 3 layers).
Observed download time and recent per-layer compute time (reported by
``PipelineDriver``) adjust the effective window:

* download-bound (download >= compute): keep the full window so the next layer
  is already resident when compute finishes;
* compute-bound (compute > download): shrink toward two layers, which is enough
  to overlap the next fetch with the current forward pass, and optionally sleep
  a fraction of the surplus compute time before starting another fetch.

HTTP backends
-------------
aiohttp is used when importable; otherwise the exact same Range request is run
through ``urllib.request`` in ``asyncio.to_thread``.  Both paths require a 206
response, a matching ``Content-Range`` and an exact body length by default.

STATUS:
  Built: asyncio producer/consumer prefetching, N+1 lookahead, adaptive pace,
         bounded MemoryBuffer integration, Range verification, retries with
         exponential backoff + jitter, aiohttp/urllib backends, stats, explicit
         drop/refetch policy.
  TBD:   No TLS/auth beyond caller-supplied headers; no HTTP/2 multiplexing;
         one request per layer (no sub-layer chunk splitting yet).
  Packages: Python standard library only.  aiohttp is optional: both the
            aiohttp and urllib backends were exercised during verification
            (urllib while aiohttp was briefly absent, aiohttp 3.14.5 for the
            final run); psutil is optional; torch/numpy are NOT required.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import re
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .MemoryBuffer import MemoryBuffer
from .SafetensorsManifest import LayerRange, SafetensorsManifest, TensorInfo

__all__ = [
    "AsyncPrefetcher",
    "LayerPayload",
    "RangeFetchError",
]

try:  # optional dependency, never required
    import aiohttp as _aiohttp
except Exception:  # pragma: no cover - depends on environment
    _aiohttp = None

_CONTENT_RANGE_RE = re.compile(
    r"^\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*$", re.IGNORECASE
)
_USER_AGENT = "AsyncPrefetcher/0.1 (+stdlib)"


class RangeFetchError(RuntimeError):
    """Raised when an HTTP Range request cannot be satisfied exactly."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        start: int | None = None,
        end: int | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.start = start
        self.end = end
        self.status = status


@dataclass(slots=True)
class LayerPayload:
    """One downloaded layer span.

    ``start``/``end`` are absolute file offsets, with ``end`` exclusive.
    ``data`` is exactly ``end - start`` bytes.
    """

    name: str
    data: bytes
    start: int
    end: int
    download_s: float = 0.0
    fetch_started: float = 0.0
    fetch_finished: float = 0.0
    retries: int = 0
    from_cache: bool = False
    dtype: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __bytes__(self) -> bytes:
        return self.data

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "nbytes": self.nbytes,
            "download_s": self.download_s,
            "retries": self.retries,
            "from_cache": self.from_cache,
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class _LayerSpec:
    name: str
    start: int
    end: int
    dtype: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


def _derive_dtype(value: Mapping[str, Any]) -> str | None:
    dtype = value.get("dtype")
    if isinstance(dtype, str) and dtype:
        return dtype
    tensors = value.get("tensors")
    if isinstance(tensors, Sequence):
        found = {
            item.get("dtype")
            for item in tensors
            if isinstance(item, Mapping) and isinstance(item.get("dtype"), str)
        }
        if len(found) == 1:
            return next(iter(found))
    return None


def _spec_from_value(name: str, value: Any) -> _LayerSpec:
    if isinstance(value, _LayerSpec):
        return value
    if isinstance(value, LayerRange):
        return _LayerSpec(value.name, value.start, value.end, value.dtype)
    if isinstance(value, TensorInfo):
        return _LayerSpec(value.name, value.abs_start, value.abs_end, value.dtype)
    if isinstance(value, Mapping):
        start = value.get("start", value.get("abs_start"))
        end = value.get("end", value.get("abs_end"))
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"layer {name!r}: manifest entry needs integer start/end"
            )
        return _LayerSpec(name, start, end, _derive_dtype(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = value
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"layer {name!r}: range must be two integers")
        return _LayerSpec(name, start, end, None)
    raise TypeError(
        f"layer {name!r}: unsupported manifest entry type "
        f"{type(value).__name__}"
    )


def _normalize_layers(
    source: Any, manifest: SafetensorsManifest | None = None
) -> dict[str, _LayerSpec]:
    if source is None:
        if manifest is None:
            raise ValueError("a manifest or layer mapping is required")
        source = manifest
    if isinstance(source, SafetensorsManifest):
        return {
            name: _LayerSpec(name, layer.start, layer.end, layer.dtype)
            for name, layer in source.layers.items()
        }
    if not isinstance(source, Mapping) and hasattr(source, "to_dict"):
        source = source.to_dict()
    if isinstance(source, Mapping):
        candidate = source.get("layers") if "layers" in source else source
        if not isinstance(candidate, Mapping):
            raise ValueError("manifest 'layers' entry must be an object")
        result: dict[str, _LayerSpec] = {}
        for name, value in candidate.items():
            if not isinstance(name, str) or not name:
                raise ValueError("layer names must be non-empty strings")
            result[name] = _spec_from_value(name, value)
        return result
    if isinstance(source, (list, tuple)):
        result = {}
        for item in source:
            if isinstance(item, str):
                if manifest is None:
                    raise ValueError(
                        "a manifest= argument is required when passing layer names"
                    )
                if item in manifest.layers:
                    layer = manifest.layers[item]
                    result[item] = _LayerSpec(
                        item, layer.start, layer.end, layer.dtype
                    )
                elif item in manifest.tensors:
                    tensor = manifest.tensors[item]
                    result[item] = _LayerSpec(
                        item, tensor.abs_start, tensor.abs_end, tensor.dtype
                    )
                else:
                    raise KeyError(f"unknown layer/tensor {item!r}")
            elif isinstance(item, (LayerRange, TensorInfo, _LayerSpec)):
                spec = _spec_from_value(getattr(item, "name", ""), item)
                result[spec.name] = spec
            elif isinstance(item, (tuple, list)) and len(item) == 3:
                name, start, end = item
                result[str(name)] = _LayerSpec(str(name), int(start), int(end))
            else:
                raise TypeError(f"unsupported layer descriptor: {item!r}")
        return result
    raise TypeError(
        "layers_or_manifest must be a SafetensorsManifest, layer mapping, "
        "or list of layer names/descriptors"
    )


class AsyncPrefetcher:
    """Stream a safetensors file layer-by-layer with an async prefetch window.

    Parameters
    ----------
    url:
        HTTP(S) URL of the safetensors object.
    layers_or_manifest:
        A :class:`SafetensorsManifest`, the dict from
        :meth:`S3ManifestWriter.to_dict`, a mapping of layer -> range, or a
        list of layer names (requires ``manifest=``).
    buffer:
        Optional :class:`MemoryBuffer`; a bounded one is created otherwise.
    lookahead:
        Number of layers to prefetch ahead of the consumer.  The window is
        ``lookahead + 1`` (current + lookahead).
    timeout / retries / backoff:
        Per-request timeout, retry count after the first attempt, and initial
        exponential-backoff delay.
    verify_content_range:
        Require a 206 response with a matching ``Content-Range`` header.
    allow_http_200:
        Permit a 200 response only when it contains exactly the requested bytes
        (some naive servers ignore Range).  Off by default.
    """

    def __init__(
        self,
        url: str,
        layers_or_manifest: Any = None,
        buffer: MemoryBuffer | None = None,
        lookahead: int = 2,
        *,
        manifest: SafetensorsManifest | None = None,
        layer_manifest: Any = None,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.2,
        max_backoff: float = 4.0,
        headers: Mapping[str, str] | None = None,
        verify_content_range: bool = True,
        allow_http_200: bool = False,
        max_depth: int | None = None,
        rss_limit_mb: float | None = None,
    ) -> None:
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if lookahead < 0:
            raise ValueError("lookahead must be >= 0")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")

        source = layer_manifest if layer_manifest is not None else layers_or_manifest
        if source is None and manifest is not None:
            source = manifest
        self._layers = _normalize_layers(source, manifest)
        if not self._layers:
            raise ValueError("no layers to prefetch")
        for name, spec in self._layers.items():
            if spec.start < 0 or spec.end < spec.start:
                raise ValueError(
                    f"layer {name!r}: invalid range [{spec.start}, {spec.end})"
                )

        self.url = url
        self.lookahead = int(lookahead)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.max_backoff = float(max_backoff)
        self.headers = dict(headers or {})
        self.verify_content_range = bool(verify_content_range)
        self.allow_http_200 = bool(allow_http_200)

        if buffer is None:
            depth = max_depth if max_depth is not None else max(2, self.lookahead + 1)
            buffer = MemoryBuffer(max_depth=depth, rss_limit_mb=rss_limit_mb)
        self.buffer = buffer
        self._buffer_max_depth = getattr(buffer, "max_depth", None)

        self._order = list(self._layers)
        self._index = {name: i for i, name in enumerate(self._order)}
        self._consumer_idx = 0
        self._producer_idx = 0

        self._inflight: dict[str, asyncio.Task[LayerPayload]] = {}
        self._producer_task: asyncio.Task[None] | None = None
        self._consumer_event: asyncio.Event | None = None
        self._session: Any = None
        self._started = False
        self._stopping = False
        self._backend = "aiohttp" if _aiohttp is not None else "urllib"

        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._bytes_downloaded = 0
        self._layers_fetched = 0
        self._retry_count = 0
        self._failure_count = 0
        self._refetches = 0
        self._cache_hits = 0
        self._buffer_drops = 0
        self._pace_sleeps = 0
        self._pace_sleep_s = 0.0
        self._download_times: deque[float] = deque(maxlen=32)
        self._compute_times: deque[float] = deque(maxlen=32)
        self._layer_bytes: deque[int] = deque(maxlen=32)
        self._errors: deque[str] = deque(maxlen=20)

    # --------------------------------------------------------------- properties
    @property
    def layer_names(self) -> list[str]:
        return list(self._order)

    @property
    def layers(self) -> dict[str, _LayerSpec]:
        return dict(self._layers)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def backend(self) -> str:
        return self._backend

    def __len__(self) -> int:
        return len(self._layers)

    # ------------------------------------------------------------------- lifecycle
    async def start(self) -> "AsyncPrefetcher":
        """Start the producer task and (if available) the aiohttp session."""
        if self._started:
            return self
        self._started = True
        self._stopping = False
        self._started_at = time.perf_counter()
        if _aiohttp is not None:
            try:
                timeout = _aiohttp.ClientTimeout(total=self.timeout)
                self._session = _aiohttp.ClientSession(
                    timeout=timeout, headers={"User-Agent": _USER_AGENT}
                )
                self._backend = "aiohttp"
            except Exception as exc:  # pragma: no cover - defensive
                self._session = None
                self._backend = "urllib"
                self._errors.append(f"aiohttp session failed, using urllib: {exc}")
        else:
            self._backend = "urllib"
        self._consumer_event = asyncio.Event()
        self._producer_task = asyncio.create_task(
            self._producer_loop(), name="AsyncPrefetcher-producer"
        )
        return self

    async def stop(self) -> None:
        """Cancel prefetching, close the session, and leave stats intact."""
        if (
            not self._started
            and self._producer_task is None
            and not self._inflight
        ):
            return
        self._stopping = True
        if self._consumer_event is not None:
            self._consumer_event.set()
        if self._producer_task is not None:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._producer_task = None
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inflight.clear()
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()
            self._session = None
        self._started = False
        self._stopped_at = time.perf_counter()

    async def __aenter__(self) -> "AsyncPrefetcher":
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    # --------------------------------------------------------------------- consume
    async def get(self, layer_name: str) -> bytes:
        """Return the exact bytes for ``layer_name`` (re-fetching on a miss)."""
        payload = await self.get_payload(layer_name)
        return payload.data

    async def get_payload(self, layer_name: str) -> LayerPayload:
        """Return a :class:`LayerPayload` with data and timing metadata."""
        if layer_name not in self._layers:
            raise KeyError(f"unknown layer {layer_name!r}")
        await self._ensure_started()
        self._advance_consumer(self._index[layer_name])

        cached = self.buffer.get(layer_name)
        if cached is not None:
            self._cache_hits += 1
            if isinstance(cached, LayerPayload):
                return cached
            spec = self._layers[layer_name]
            return LayerPayload(
                name=layer_name,
                data=bytes(cached),
                start=spec.start,
                end=spec.end,
                download_s=0.0,
                from_cache=True,
                dtype=spec.dtype,
            )
        self._refetches += 1
        return await self._fetch_payload(layer_name)

    def evict(self, layer_name: str) -> bool:
        """Drop a layer from the shared buffer (used by the driver after use)."""
        return self.buffer.evict(layer_name)

    def record_compute_time(self, seconds: float) -> None:
        """Feed the observed per-layer compute time into the pace heuristic."""
        if seconds < 0:
            raise ValueError("compute time must be non-negative")
        self._compute_times.append(float(seconds))

    # --------------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        download_avg = (
            sum(self._download_times) / len(self._download_times)
            if self._download_times
            else 0.0
        )
        compute_avg = (
            sum(self._compute_times) / len(self._compute_times)
            if self._compute_times
            else 0.0
        )
        bytes_avg = (
            sum(self._layer_bytes) / len(self._layer_bytes)
            if self._layer_bytes
            else 0.0
        )
        total_download = sum(self._download_times)
        bandwidth_mbps = (
            (self._bytes_downloaded * 8.0 / 1_000_000.0) / total_download
            if total_download > 0
            else 0.0
        )
        return {
            "backend": self._backend,
            "url": self.url,
            "started": self._started,
            "layers_total": len(self._layers),
            "layers_fetched": self._layers_fetched,
            "bytes_downloaded": self._bytes_downloaded,
            "retries": self._retry_count,
            "failures": self._failure_count,
            "refetches": self._refetches,
            "cache_hits": self._cache_hits,
            "buffer_drops": self._buffer_drops,
            "avg_download_s": download_avg,
            "avg_compute_s": compute_avg,
            "avg_layer_bytes": bytes_avg,
            "bandwidth_mbps": bandwidth_mbps,
            "lookahead": self.lookahead,
            "effective_lookahead": self._effective_lookahead(),
            "pace_sleeps": self._pace_sleeps,
            "pace_sleep_s": self._pace_sleep_s,
            "in_flight": len(self._inflight),
            "consumer_index": self._consumer_idx,
            "producer_index": self._producer_idx,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "errors": list(self._errors),
            "buffer": self.buffer.stats(),
        }

    # ------------------------------------------------------------- producer logic
    async def _ensure_started(self) -> None:
        if not self._started:
            await self.start()

    def _advance_consumer(self, index: int) -> None:
        if index > self._consumer_idx:
            self._consumer_idx = index
            if self._producer_idx < index:
                self._producer_idx = index
            if self._consumer_event is not None:
                self._consumer_event.set()

    def _effective_lookahead(self) -> int:
        base = self.lookahead + 1
        if base <= 2:
            return base
        if not self._compute_times or not self._download_times:
            return base
        compute = sum(self._compute_times) / len(self._compute_times)
        download = sum(self._download_times) / len(self._download_times)
        if compute <= 0 or download <= 0 or compute <= download:
            effective = base
        else:
            # Compute-bound: two resident layers already overlap the next fetch
            # with the current forward pass; extra depth mostly wastes memory.
            scaled = 2 + int((download / compute) * max(0, base - 2))
            effective = max(2, min(base, scaled))
        if self._buffer_max_depth is not None:
            effective = min(effective, max(1, int(self._buffer_max_depth)))
        return max(1, effective)

    def _pace_delay(self) -> float:
        if not self._compute_times or not self._download_times:
            return 0.0
        compute = sum(self._compute_times) / len(self._compute_times)
        download = sum(self._download_times) / len(self._download_times)
        if compute <= download or download <= 0:
            return 0.0
        if self._producer_idx <= self._consumer_idx:
            return 0.0
        if len(self.buffer) < max(1, self._effective_lookahead() - 1):
            return 0.0
        # Bounded cosmetic pause: avoid a burst of fetches that will just sit in
        # the buffer while the consumer computes.  Never delays the next layer.
        return min(0.02, max(0.0, (compute - download) * 0.25))

    async def _producer_loop(self) -> None:
        try:
            while not self._stopping and self._producer_idx < len(self._order):
                window = self._effective_lookahead()
                # Launch as many independent Range requests as the window and
                # in-flight cap allow.  This is what actually overlaps downloads
                # with the driver's compute (the producer is not serialized on
                # one request).
                while (
                    not self._stopping
                    and self._producer_idx < len(self._order)
                    and len(self._inflight) < max(1, window)
                    and self._producer_idx - self._consumer_idx < window
                ):
                    name = self._order[self._producer_idx]
                    if name in self.buffer or name in self._inflight:
                        self._producer_idx += 1
                        continue
                    delay = self._pace_delay()
                    if delay > 0:
                        self._pace_sleeps += 1
                        self._pace_sleep_s += delay
                        await asyncio.sleep(delay)
                        if self._stopping:
                            break
                    self._start_fetch(name)
                    self._producer_idx += 1

                if self._stopping or self._producer_idx >= len(self._order):
                    break

                # Wait for a fetch to finish, the consumer to advance, or a
                # short timeout.  Never spin on the event loop.
                waiters = list(self._inflight.values())
                event = self._consumer_event
                event_task: asyncio.Task[bool] | None = None
                if event is not None:
                    event.clear()
                    if (
                        self._producer_idx - self._consumer_idx >= window
                        or len(self._inflight) >= max(1, window)
                    ):
                        event_task = asyncio.create_task(event.wait())
                if not waiters and event_task is None:
                    await asyncio.sleep(0.005)
                    continue
                tasks: list[asyncio.Future[Any]] = list(waiters)
                if event_task is not None:
                    tasks.append(event_task)
                try:
                    await asyncio.wait(
                        tasks,
                        timeout=0.2,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    if event_task is not None:
                        event_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await event_task
        except asyncio.CancelledError:
            raise

    # ---------------------------------------------------------------- fetch logic
    def _start_fetch(self, name: str) -> "asyncio.Task[LayerPayload]":
        task = self._inflight.get(name)
        if task is None:
            task = asyncio.create_task(self._fetch_impl(name), name=f"fetch:{name}")
            self._inflight[name] = task
            task.add_done_callback(
                lambda finished, layer=name: self._on_fetch_done(layer, finished)
            )
        return task

    def _fetch_payload(self, name: str) -> "asyncio.Future[LayerPayload]":
        return asyncio.shield(self._start_fetch(name))

    def _on_fetch_done(self, name: str, task: asyncio.Task[LayerPayload]) -> None:
        if self._inflight.get(name) is task:
            self._inflight.pop(name, None)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception:  # pragma: no cover - defensive
            exc = None
        if exc is not None:
            self._errors.append(f"fetch {name}: {exc}")

    async def _fetch_impl(self, name: str) -> LayerPayload:
        spec = self._layers[name]
        if spec.length == 0:
            payload = LayerPayload(
                name=name,
                data=b"",
                start=spec.start,
                end=spec.end,
                dtype=spec.dtype,
            )
            if not self.buffer.put(name, payload):
                self._buffer_drops += 1
            return payload

        started = time.perf_counter()
        last_error: Exception | None = None
        attempt = 0
        for attempt in range(self.retries + 1):
            try:
                data = await self._http_get(spec)
                if len(data) != spec.length:
                    raise RangeFetchError(
                        f"layer {name!r}: expected {spec.length} bytes, "
                        f"received {len(data)}",
                        url=self.url,
                        start=spec.start,
                        end=spec.end,
                    )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    self._failure_count += 1
                    raise RangeFetchError(
                        f"layer {name!r}: range fetch failed after "
                        f"{attempt + 1} attempts: {exc}",
                        url=self.url,
                        start=spec.start,
                        end=spec.end,
                    ) from exc
                self._retry_count += 1
                delay = min(
                    self.max_backoff,
                    self.backoff * (2 ** attempt),
                )
                delay *= 0.5 + random.random() * 0.5
                await asyncio.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise RangeFetchError(
                f"layer {name!r}: no data", url=self.url,
                start=spec.start, end=spec.end
            )

        finished = time.perf_counter()
        payload = LayerPayload(
            name=name,
            data=data,
            start=spec.start,
            end=spec.end,
            download_s=finished - started,
            fetch_started=started,
            fetch_finished=finished,
            retries=attempt,
            dtype=spec.dtype,
        )
        if not self.buffer.put(name, payload):
            self._buffer_drops += 1
        self._layers_fetched += 1
        self._bytes_downloaded += len(data)
        self._download_times.append(payload.download_s)
        self._layer_bytes.append(len(data))
        return payload

    # ------------------------------------------------------------------- HTTP
    async def _http_get(self, spec: _LayerSpec) -> bytes:
        if _aiohttp is not None and self._session is not None:
            return await self._aiohttp_get(spec)
        return await asyncio.to_thread(self._urllib_get, spec)

    def _request_headers(self, spec: _LayerSpec) -> dict[str, str]:
        headers = {
            "Range": f"bytes={spec.start}-{spec.end - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": _USER_AGENT,
        }
        headers.update(self.headers)
        return headers

    def _validate_response(
        self,
        spec: _LayerSpec,
        status: int,
        content_range: str | None,
        body_length: int,
    ) -> None:
        if status != 206:
            if self.allow_http_200 and status == 200:
                if body_length != spec.length:
                    raise RangeFetchError(
                        f"server returned 200 with {body_length} bytes; "
                        f"requested {spec.length}",
                        url=self.url, start=spec.start, end=spec.end, status=status,
                    )
            else:
                raise RangeFetchError(
                    f"expected HTTP 206, got {status}",
                    url=self.url, start=spec.start, end=spec.end, status=status,
                )
        if status == 206 and self.verify_content_range:
            if not content_range:
                raise RangeFetchError(
                    "206 response is missing Content-Range",
                    url=self.url, start=spec.start, end=spec.end, status=status,
                )
            match = _CONTENT_RANGE_RE.match(content_range)
            if not match:
                raise RangeFetchError(
                    f"malformed Content-Range {content_range!r}",
                    url=self.url, start=spec.start, end=spec.end, status=status,
                )
            got_start, got_end = int(match.group(1)), int(match.group(2))
            expected_end = spec.end - 1
            if got_start != spec.start or got_end != expected_end:
                raise RangeFetchError(
                    f"Content-Range {content_range!r} does not match requested "
                    f"bytes={spec.start}-{expected_end}",
                    url=self.url, start=spec.start, end=spec.end, status=status,
                )
            total = match.group(3)
            if total != "*" and int(total) <= got_end:
                raise RangeFetchError(
                    f"Content-Range total {total} is not beyond end {got_end}",
                    url=self.url, start=spec.start, end=spec.end, status=status,
                )
        if body_length != spec.length:
            raise RangeFetchError(
                f"expected {spec.length} body bytes, got {body_length}",
                url=self.url, start=spec.start, end=spec.end, status=status,
            )

    async def _aiohttp_get(self, spec: _LayerSpec) -> bytes:
        assert _aiohttp is not None
        async with self._session.get(
            self.url, headers=self._request_headers(spec)
        ) as response:
            body = await response.read()
            self._validate_response(
                spec,
                response.status,
                response.headers.get("Content-Range"),
                len(body),
            )
            return body

    def _urllib_get(self, spec: _LayerSpec) -> bytes:
        request = urllib.request.Request(
            self.url, headers=self._request_headers(spec)
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                body = response.read(spec.length + 1)
                content_range = response.headers.get("Content-Range")
        except urllib.error.HTTPError as exc:
            with contextlib.suppress(Exception):
                exc.close()
            raise RangeFetchError(
                f"HTTP {exc.code} for bytes={spec.start}-{spec.end - 1}",
                url=self.url, start=spec.start, end=spec.end, status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise RangeFetchError(
                f"network error for bytes={spec.start}-{spec.end - 1}: {exc}",
                url=self.url, start=spec.start, end=spec.end,
            ) from exc
        if len(body) > spec.length:
            raise RangeFetchError(
                f"server sent more than {spec.length} bytes",
                url=self.url, start=spec.start, end=spec.end, status=status,
            )
        self._validate_response(spec, status, content_range, len(body))
        return body

    def __repr__(self) -> str:
        return (
            f"AsyncPrefetcher(layers={len(self._layers)}, lookahead={self.lookahead}, "
            f"backend={self._backend}, started={self._started})"
        )
