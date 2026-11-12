#!/usr/bin/env python3
"""TTFT / I/O-masking / RSS benchmark harness.

STATUS:
  BUILT: end-to-end harness comparing (a) monolithic baseline (download whole
         file, then compute every layer) against (b) progressive layer
         streaming (AsyncPrefetcher + PipelineDriver contract from src/core,
         with a stdlib fallback engine when src/core is absent).  Measures
         TTFT, end-to-end latency, blocked I/O wait, compute time, I/O masking
         ratio, bytes downloaded, peak RSS and an RSS trajectory.  Writes
         results/benchmark.csv and per-layer detail, then prints
         mean/median/p95/min/max per mode.  Verified end-to-end against
         src/core on the synthetic 8-layer model at 8 Mbit/s (see
         results/benchmark.csv).  If no run succeeds it prints
         "TBD — pending successful run" and writes a TBD row; it never
         invents numbers.
  TBD:   validation against production MinIO/S3 + ``tc`` and a real
         transformer checkpoint.  The recorded numbers are for a synthetic F32
         model over HTTP loopback and must not be generalised.
  PACKAGES: aiohttp>=3.9 (only if src/core uses it), psutil>=5.9 (RSS),
         requests>=2.31 (declared for parity; harness uses stdlib http.client).

INTEGRATION:
  Contract used from src/core (see docs/build_todo.md):
      SafetensorsManifest(path).layers[name].start/end
      AsyncPrefetcher(url, layers, buffer=None, lookahead=2)
          await start(); await get(name); await stop()
      MemoryBuffer(max_depth, rss_limit_mb)
      PipelineDriver(prefetcher, compute_engine=None)
          metrics = await run(layer_names)
  If src/core is missing, ``--engine core`` raises an actionable error; the
  default ``--engine auto`` uses the fallback engine and labels every row.

HONESTY / SUBSTITUTION NOTE:
  The local RangeServer + HTTP loopback is a *stand-in* for MinIO/S3 + Linux
  ``tc``/``netem``.  The server throttle is shared across concurrent range
  requests, modelling one shaped client link.  Production path: MinIO object
  store + ``tc`` on the worker NIC, AsyncPrefetcher pointed at MinIO.
  The baseline pre-allocates the full file, so peak-RSS is *measured and
  reported*, never assumed to improve.  CPU layer compute can dominate the
  network; measured masking is reported as-is (do not expect >70% on CPU).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import http.client
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import psutil
except ImportError:  # pragma: no cover - setup.sh installs it
    psutil = None  # type: ignore[assignment]

# Make ``src.*`` importable when the harness is invoked from another cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # package import
    from ..storage.safetensors_ranges import parse_safetensors
    from ..storage.RangeServer import RangeServer
except ImportError:  # direct execution
    from storage.safetensors_ranges import parse_safetensors  # type: ignore
    from storage.RangeServer import RangeServer  # type: ignore


CSV_COLUMNS = [
    "run",
    "mode",
    "ttft_s",
    "first_layer_s",
    "total_s",
    "io_wait_s",
    "compute_s",
    "masking_ratio",
    "peak_rss_mb",
    "bytes_downloaded",
    "bandwidth_mbps",
    "model",
    "notes",
]

PER_LAYER_COLUMNS = [
    "run",
    "mode",
    "layer",
    "bytes",
    "fetch_s",
    "compute_s",
    "blocked_s",
    "fetch_start_s",
    "fetch_end_s",
    "compute_start_s",
    "compute_end_s",
]

MIB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rss_mb() -> float:
    """Current process RSS in MiB (psutil when available, /proc otherwise)."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / MIB
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / MIB
    except Exception:  # noqa: BLE001
        return float("nan")


class RSSSampler(threading.Thread):
    """Sample RSS every ``interval`` seconds in a daemon thread."""

    def __init__(self, interval: float = 0.02):
        super().__init__(name="RSSSampler", daemon=True)
        self.interval = max(0.005, float(interval))
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self.t0 = time.perf_counter()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter() - self.t0, _rss_mb()))
            except Exception:  # noqa: BLE001
                break
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.0)

    def peak(self) -> float:
        values = [r for _, r in self.samples if not math.isnan(r)]
        return max(values) if values else float("nan")


def _downsample(samples: list[tuple[float, float]], max_points: int = 400) -> list[tuple[float, float]]:
    if len(samples) <= max_points:
        return list(samples)
    stride = math.ceil(len(samples) / max_points)
    reduced = samples[::stride]
    if reduced[-1] != samples[-1]:
        reduced.append(samples[-1])
    return reduced


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, str):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "TBD"
    return f"{f:.{digits}f}"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; requests is declared but not needed here)
# ---------------------------------------------------------------------------
def http_get(
    url: str,
    range_header: str | None = None,
    timeout: float = 900.0,
) -> tuple[int, bytes, dict[str, str]]:
    parts = urlsplit(url)
    if parts.scheme not in ("http", ""):
        raise ValueError(f"only http:// URLs are supported, got {url!r}")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Range": range_header} if range_header else {}
    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        body = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
    finally:
        conn.close()
    return status, body, resp_headers


def fetch_layer(url: str, start: int, end_exclusive: int, timeout: float = 900.0) -> bytes:
    """Fetch ``[start, end_exclusive)`` with a single HTTP Range request."""
    if end_exclusive <= start:
        return b""
    range_header = f"bytes={start}-{end_exclusive - 1}"
    status, body, headers = http_get(url, range_header, timeout=timeout)
    if status != 206:
        raise RuntimeError(
            f"range request {range_header} expected HTTP 206, got {status} "
            f"(Content-Range={headers.get('content-range')!r})"
        )
    expected = end_exclusive - start
    if len(body) != expected:
        raise RuntimeError(
            f"range request {range_header} returned {len(body)} bytes, expected {expected}"
        )
    return body


# ---------------------------------------------------------------------------
# Synthetic compute engine (stand-in for transformer layer GEMMs)
# ---------------------------------------------------------------------------
class SyntheticComputeEngine:
    """Deterministic pure-Python MAC loop used as a CPU compute stand-in.

    ``work_factor`` scales the number of multiply-add iterations.  The loop is
    deliberately CPU-bound so that network/compute overlap is observable; it is
    NOT a transformer and makes no numerical claim.  ``macs_per_byte=7.5`` is
    calibrated so ~1 MiB of payload costs ~0.5 s on the development box, i.e.
    about half the serial transfer time at 8 Mbit/s (1 MiB/s).  This keeps the
    default benchmark I/O-bound so measured masking lands in a defensible
    range; tune with ``--work-factor`` for other bandwidths.  The chosen factor
    is reported in the harness header.
    """

    def __init__(self, work_factor: float = 1.0, macs_per_byte: float = 7.5):
        self.work_factor = max(0.0, float(work_factor))
        self.macs_per_byte = max(0.0, float(macs_per_byte))

    def run_layer(
        self,
        name: str,
        payload: bytes,
        *,
        dtype: str | None = None,  # accepted for PipelineDriver compatibility
        work_factor: float | None = None,
        **_kwargs: Any,
    ) -> float:
        wf = self.work_factor if work_factor is None else max(0.0, float(work_factor))
        iterations = int(len(payload) * self.macs_per_byte * wf)
        acc = 0.0
        x = 1.0000001
        for i in range(iterations):
            acc = acc * x + (i & 7)
        return acc

    # Compatibility aliases for whatever PipelineDriver expects.
    __call__ = run_layer
    compute_layer = run_layer
    forward = run_layer


# ---------------------------------------------------------------------------
# Fallback streaming engine (only used when src/core is unavailable)
# ---------------------------------------------------------------------------
class FallbackPrefetcher:
    """Minimal lookahead range prefetcher implementing the core contract.

    Fetches run in worker threads via ``asyncio.to_thread`` so they overlap the
    (threaded) synthetic compute.  ``lookahead`` is the scheduling depth
    *including* the layer currently being consumed, so ``lookahead=2`` means
    "start fetching the next layer as soon as the current one is available".
    On a shared link this prioritises layer 0 (TTFT) instead of splitting the
    link across every future layer.  (A production prefetcher can raise the
    depth when bandwidth is plentiful.)
    """

    def __init__(self, url: str, layers: dict[str, dict], lookahead: int = 2, timeout: float = 900.0):
        self.url = url
        self.layers = layers
        self.names = list(layers)
        self.lookahead = max(1, int(lookahead))
        self.timeout = timeout
        self._tasks: dict[str, asyncio.Task] = {}
        self._scheduled: set[str] = set()
        self.fetch_times: dict[str, tuple[float, float, float, int]] = {}
        self.total_bytes = 0

    async def start(self) -> None:
        # Start only the first layer so it gets the full shared link.
        if self.names:
            self._schedule(0)

    def _schedule(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.names):
            return
        name = self.names[idx]
        if name in self._scheduled:
            return
        self._scheduled.add(name)
        self._tasks[name] = asyncio.create_task(self._fetch(name))

    async def _fetch(self, name: str) -> bytes:
        meta = self.layers[name]
        start = int(meta["start"])
        end = int(meta["end"])
        t_start = time.perf_counter()
        data = await asyncio.to_thread(fetch_layer, self.url, start, end, self.timeout)
        t_end = time.perf_counter()
        self.fetch_times[name] = (t_start, t_end, t_end - t_start, len(data))
        self.total_bytes += len(data)
        return data

    async def get(self, name: str) -> bytes:
        idx = self.names.index(name)
        self._schedule(idx)
        data = await self._tasks[name]
        # Keep at most (lookahead - 1) layers in flight ahead of the current
        # one; this preserves layer-0 priority on a shared throttled link.
        ahead = max(0, self.lookahead - 1)
        for nxt in range(idx + 1, min(idx + 1 + ahead, len(self.names))):
            self._schedule(nxt)
        return data

    async def stop(self) -> None:
        pending = [t for t in self._tasks.values() if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class FallbackDriver:
    """Layer-by-layer driver with per-layer overlap instrumentation."""

    def __init__(
        self,
        prefetcher: FallbackPrefetcher,
        compute_engine: SyntheticComputeEngine,
        names: list[str],
        bandwidth_mbps: float,
    ):
        self.prefetcher = prefetcher
        self.compute_engine = compute_engine
        self.names = names
        self.bandwidth_mbps = float(bandwidth_mbps)

    async def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        ttft: float | None = None
        first_layer: float | None = None
        io_wait = 0.0
        compute_total = 0.0
        per_layer: list[dict[str, Any]] = []

        for i, name in enumerate(self.names):
            want = time.perf_counter()
            payload = await self.prefetcher.get(name)
            ready = time.perf_counter()
            blocked = max(0.0, ready - want)
            io_wait += blocked

            c_start = time.perf_counter()
            await asyncio.to_thread(self.compute_engine.run_layer, name, payload)
            c_end = time.perf_counter()
            compute_total += c_end - c_start
            if i == 0:
                first_layer = c_end - t0

            ft = self.prefetcher.fetch_times.get(name)
            per_layer.append(
                {
                    "layer": name,
                    "bytes": len(payload),
                    "fetch_s": ft[2] if ft else None,
                    "compute_s": c_end - c_start,
                    "blocked_s": blocked,
                    "fetch_start_s": (ft[0] - t0) if ft else None,
                    "fetch_end_s": (ft[1] - t0) if ft else None,
                    "compute_start_s": c_start - t0,
                    "compute_end_s": c_end - t0,
                }
            )

        total = time.perf_counter() - t0
        ttft = total  # first token requires the full forward pass
        nbytes = self.prefetcher.total_bytes
        nominal_io = self._nominal_io(nbytes, per_layer)
        masking = _clamp01(1.0 - io_wait / nominal_io) if nominal_io > 0 else 0.0
        return {
            "ttft_s": ttft,
            "first_layer_s": first_layer,
            "total_s": total,
            "io_wait_s": io_wait,
            "compute_s": compute_total,
            "masking_ratio": masking,
            "bytes_downloaded": nbytes,
            "nominal_io_s": nominal_io,
            "per_layer": per_layer,
        }

    def _nominal_io(self, nbytes: int, per_layer: list[dict[str, Any]]) -> float:
        if self.bandwidth_mbps > 0:
            return nbytes / (self.bandwidth_mbps * 1_000_000.0 / 8.0)
        total = 0.0
        for row in per_layer:
            if row.get("fetch_s") is not None:
                total += float(row["fetch_s"])
        return total


# ---------------------------------------------------------------------------
# Core integration (src/core) — code defensively against a moving contract
# ---------------------------------------------------------------------------
def core_import_error() -> str | None:
    """Return None if the core modules import, else a human-readable reason."""
    try:
        from src.core.AsyncPrefetcher import AsyncPrefetcher  # noqa: F401
        from src.core.PipelineDriver import PipelineDriver  # noqa: F401
        from src.core.SafetensorsManifest import SafetensorsManifest  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def load_manifest(path: str) -> tuple[dict[str, Any], str]:
    """Prefer ``src.core.SafetensorsManifest``; fall back to the stdlib parser."""
    if core_import_error() is None:
        try:
            from src.core.SafetensorsManifest import SafetensorsManifest

            manifest = SafetensorsManifest(path)
            layers: dict[str, dict[str, Any]] = {}
            for name, layer in manifest.layers.items():
                start = int(layer.start)
                end = int(layer.end)
                layers[name] = {"start": start, "end": end, "length": end - start}
            if layers:
                return (
                    {
                        "file": os.path.abspath(path),
                        "size": int(getattr(manifest, "file_size", os.path.getsize(path))),
                        "layers": layers,
                    },
                    "core",
                )
        except Exception as exc:  # noqa: BLE001 - fall back honestly
            print(
                f"WARNING: src.core.SafetensorsManifest failed ({type(exc).__name__}: "
                f"{exc}); falling back to stdlib parser.",
                file=sys.stderr,
            )
    return parse_safetensors(path), "stdlib"


def _first_number(mapping: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalise_per_layer(raw: Any, names: list[str]) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        items = [(str(k), v) for k, v in raw.items()]
    elif isinstance(raw, list):
        items = [(str(i), v) for i, v in enumerate(raw)]
    else:
        return []

    rows: list[dict[str, Any]] = []
    for idx, (key, value) in enumerate(items):
        if not isinstance(value, dict):
            continue
        name = str(value.get("layer") or value.get("name") or key)
        if name.isdigit() and int(name) < len(names):
            name = names[int(name)]
        rows.append(
            {
                "layer": name,
                "bytes": value.get("bytes") or value.get("nbytes") or value.get("length"),
                "fetch_s": _first_number(value, ["fetch_s", "fetch_time_s", "download_s", "io_s"]),
                "compute_s": _first_number(value, ["compute_s", "compute_time_s", "forward_s"]),
                "blocked_s": _first_number(value, ["blocked_s", "io_wait_s", "wait_s"]),
                "fetch_start_s": _first_number(value, ["fetch_start_s", "fetch_start", "download_start_s"]),
                "fetch_end_s": _first_number(value, ["fetch_end_s", "fetch_end", "download_end_s"]),
                "compute_start_s": _first_number(value, ["compute_start_s", "compute_start"]),
                "compute_end_s": _first_number(value, ["compute_end_s", "compute_end", "end_s"]),
            }
        )
    return rows


async def _run_pipeline_core(cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Run the real src/core pipeline. Raises an actionable error on mismatch."""
    from src.core.AsyncPrefetcher import AsyncPrefetcher
    from src.core.PipelineDriver import PipelineDriver

    buffer = None
    try:
        from src.core.MemoryBuffer import MemoryBuffer

        try:
            buffer = MemoryBuffer(
                max_depth=max(2, int(cfg["lookahead"]) + 1),
                rss_limit_mb=cfg.get("rss_limit_mb", 1024),
            )
        except TypeError:
            buffer = MemoryBuffer()
    except Exception:  # noqa: BLE001 - buffer is optional in the contract
        buffer = None

    try:
        prefetcher = AsyncPrefetcher(
            cfg["url"], cfg["layers"], buffer=buffer, lookahead=cfg["lookahead"]
        )
    except TypeError:
        prefetcher = AsyncPrefetcher(cfg["url"], cfg["layers"], lookahead=cfg["lookahead"])

    engine = SyntheticComputeEngine(cfg["work_factor"])
    await prefetcher.start()
    driver = PipelineDriver(prefetcher, compute_engine=engine)
    t0 = time.perf_counter()
    try:
        try:
            metrics = await driver.run(names)
        except TypeError:
            # Older/alternate contract: run() with no arguments.
            metrics = await driver.run()
    finally:
        try:
            await prefetcher.stop()
        except Exception:  # noqa: BLE001
            pass
    elapsed = time.perf_counter() - t0

    m = metrics if isinstance(metrics, dict) else {}
    per_layer = _normalise_per_layer(m.get("per_layer") or m.get("layers"), names)

    total = _first_number(m, ["total_s", "total", "e2e_s", "latency_s", "elapsed_s"])
    if total is None:
        total = elapsed

    # TTFT is defined as completion of the full forward pass (the first token
    # needs every layer).  first_layer_s is reported separately as streaming
    # fill latency and is NOT used as TTFT.
    first_layer = _first_number(m, ["first_layer_s", "first_layer", "layer0_s"])
    if first_layer is None and per_layer:
        if per_layer[0].get("compute_end_s") is not None:
            first_layer = float(per_layer[0]["compute_end_s"])
    ttft = total

    io_wait = _first_number(m, ["io_wait_s", "io_wait", "blocked_io_s", "blocked_s"])
    compute = _first_number(m, ["compute_s", "compute_time_s", "compute_total_s"])

    nbytes = _first_number(m, ["bytes_downloaded", "bytes", "nbytes"])
    if nbytes is None:
        nbytes = getattr(prefetcher, "total_bytes", None)
    if nbytes is None:
        nbytes = sum(int(cfg["layers"][n]["end"]) - int(cfg["layers"][n]["start"]) for n in names)

    nominal_io = None
    if cfg["bandwidth_mbps"] > 0:
        nominal_io = float(nbytes) / (cfg["bandwidth_mbps"] * 1_000_000.0 / 8.0)

    # Use ONE masking definition across baseline and pipeline so the CSV is
    # comparable: masking_ratio = 1 - io_wait / nominal_io, where nominal_io is
    # the serial time to transfer the measured bytes at the configured rate.
    # (Core's own masking_ratio uses compute/(compute+io_wait); we do not mix
    # definitions.  Its io_masking_ratio is only a last-resort fallback.)
    masking: float | None = None
    if io_wait is not None and nominal_io:
        masking = _clamp01(1.0 - io_wait / nominal_io)
    if masking is None:
        masking = _first_number(m, ["io_masking_ratio", "masking_ratio"])
    if compute is None and per_layer:
        vals = [float(r["compute_s"]) for r in per_layer if r.get("compute_s") is not None]
        compute = sum(vals) if vals else None
    if io_wait is None and per_layer:
        vals = [float(r["blocked_s"]) for r in per_layer if r.get("blocked_s") is not None]
        io_wait = sum(vals) if vals else None

    if ttft is None:
        raise RuntimeError(
            "src/core PipelineDriver returned no TTFT metric (keys="
            f"{sorted(m) if m else 'none'}). INTEGRATION: adapt "
            "BenchmarkHarness._run_pipeline_core to the actual metrics dict."
        )

    return {
        "ttft_s": ttft,
        "first_layer_s": first_layer,
        "total_s": total,
        "io_wait_s": io_wait,
        "compute_s": compute,
        "masking_ratio": masking,
        "bytes_downloaded": int(nbytes),
        "nominal_io_s": nominal_io,
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Baseline (monolithic) run
# ---------------------------------------------------------------------------
def _run_baseline(cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    engine = SyntheticComputeEngine(cfg["work_factor"])
    t0 = time.perf_counter()
    status, body, headers = http_get(cfg["url"], timeout=cfg["timeout"])
    download_s = time.perf_counter() - t0
    if status != 200:
        raise RuntimeError(
            f"baseline full GET expected HTTP 200, got {status} "
            f"(Content-Range={headers.get('content-range')!r})"
        )
    if len(body) != int(cfg["size"]):
        raise RuntimeError(
            f"baseline downloaded {len(body)} bytes, expected {cfg['size']}"
        )

    ttft: float | None = None
    first_layer: float | None = None
    compute_total = 0.0
    per_layer: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        meta = cfg["layers"][name]
        payload = body[int(meta["start"]): int(meta["end"])]
        c_start = time.perf_counter()
        engine.run_layer(name, payload)
        c_end = time.perf_counter()
        compute_total += c_end - c_start
        if i == 0:
            first_layer = c_end - t0
        per_layer.append(
            {
                "layer": name,
                "bytes": len(payload),
                "fetch_s": None,
                "compute_s": c_end - c_start,
                "blocked_s": None,
                "fetch_start_s": None,
                "fetch_end_s": None,
                "compute_start_s": c_start - t0,
                "compute_end_s": c_end - t0,
            }
        )

    total = time.perf_counter() - t0
    ttft = total  # first token requires the full forward pass
    nbytes = len(body)
    nominal_io = (
        nbytes / (cfg["bandwidth_mbps"] * 1_000_000.0 / 8.0)
        if cfg["bandwidth_mbps"] > 0
        else download_s
    )
    io_wait = download_s
    masking = _clamp01(1.0 - io_wait / nominal_io) if nominal_io > 0 else 0.0
    return {
        "ttft_s": ttft,
        "first_layer_s": first_layer,
        "total_s": total,
        "io_wait_s": io_wait,
        "compute_s": compute_total,
        "masking_ratio": masking,
        "bytes_downloaded": nbytes,
        "nominal_io_s": nominal_io,
        "per_layer": per_layer,
    }


async def _run_pipeline_fallback(cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    engine = SyntheticComputeEngine(cfg["work_factor"])
    prefetcher = FallbackPrefetcher(
        cfg["url"], cfg["layers"], lookahead=cfg["lookahead"], timeout=cfg["timeout"]
    )
    await prefetcher.start()
    driver = FallbackDriver(prefetcher, engine, names, cfg["bandwidth_mbps"])
    try:
        return await driver.run()
    finally:
        await prefetcher.stop()


# ---------------------------------------------------------------------------
# Worker executed in a fresh spawned process (honest per-run RSS)
# ---------------------------------------------------------------------------
def _run_one(cfg: dict[str, Any]) -> dict[str, Any]:
    names = list(cfg["layers"])
    engine_choice = cfg["engine"]
    if engine_choice == "auto":
        err = core_import_error()
        engine_choice = "fallback" if err else "core"
    if engine_choice == "core":
        err = core_import_error()
        if err:
            raise RuntimeError(
                "src/core is not importable, so --engine core cannot run. "
                f"Reason: {err}. INTEGRATION: wait for WS-2 to land "
                "src/core/{SafetensorsManifest,AsyncPrefetcher,MemoryBuffer,"
                "PipelineDriver}.py, or use --engine fallback for the "
                "self-contained benchmark."
            )

    sampler = RSSSampler(interval=cfg["rss_interval"])
    sampler.start()
    try:
        if cfg["mode"] == "baseline":
            metrics = _run_baseline(cfg, names)
        elif engine_choice == "core":
            metrics = asyncio.run(_run_pipeline_core(cfg, names))
        else:
            metrics = asyncio.run(_run_pipeline_fallback(cfg, names))
    finally:
        sampler.stop()

    metrics["peak_rss_mb"] = sampler.peak()
    metrics["rss_trace"] = _downsample(sampler.samples, cfg.get("rss_points", 400))
    metrics["engine"] = engine_choice
    return metrics


def _run_worker(cfg: dict[str, Any], result_queue: Any) -> None:
    try:
        result_queue.put({"ok": True, "result": _run_one(cfg)})
    except BaseException as exc:  # noqa: BLE001 - must report failures to parent
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_in_subprocess(cfg: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_run_worker, args=(cfg, result_queue), daemon=True)
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "ok": False,
            "error": f"worker timed out after {timeout_s:.0f}s",
            "traceback": "",
        }
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "ok": False,
            "error": f"worker exited rc={proc.exitcode} without a result",
            "traceback": "",
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _write_per_layer(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_LAYER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PER_LAYER_COLUMNS})


def _print_stats(rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("ttft_s", 4),
        ("first_layer_s", 4),
        ("total_s", 4),
        ("masking_ratio", 4),
        ("peak_rss_mb", 2),
    ]
    modes = []
    for row in rows:
        if row["mode"] not in modes:
            modes.append(row["mode"])

    print("\n=== Summary (measured; no fabricated values) ===")
    header = f"{'mode':<10} {'metric':<16} {'mean':>10} {'median':>10} {'p95':>10} {'min':>10} {'max':>10} {'n':>4}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        for metric, digits in metrics:
            values: list[float] = []
            for r in mode_rows:
                try:
                    values.append(float(r[metric]))
                except (KeyError, TypeError, ValueError):
                    continue
            if not values:
                print(f"{mode:<10} {metric:<16} {'TBD':>10}")
                continue
            print(
                f"{mode:<10} {metric:<16} "
                f"{statistics.fmean(values):>{10}.{digits}f} "
                f"{statistics.median(values):>{10}.{digits}f} "
                f"{_percentile(values, 0.95):>{10}.{digits}f} "
                f"{min(values):>{10}.{digits}f} "
                f"{max(values):>{10}.{digits}f} "
                f"{len(values):>4}"
            )
    print(
        "\nDefinitions: TTFT = request start -> completion of the full forward pass "
        "(first token requires every layer); first_layer = request start -> layer-0 "
        "compute done (streaming fill, NOT an LLM TTFT); total = request start -> "
        "last layer output; io_wait = sum of time the driver was blocked on a layer; "
        "masking_ratio = 1 - io_wait / nominal_io, nominal_io = bytes / (bandwidth/8)."
    )
    print(
        "Substitution: local RangeServer + HTTP loopback stands in for MinIO/S3 + tc. "
        "Production path is MinIO + tc/netem shaping. Peak RSS is measured, not assumed."
    )


def _print_failures(failures: list[dict[str, Any]]) -> None:
    if not failures:
        return
    print("\n=== Failed runs (excluded from statistics) ===")
    for f in failures:
        print(f"  run={f['run']} mode={f['mode']} engine={f['engine']}: {f['error']}")
        if f.get("traceback"):
            for line in str(f["traceback"]).rstrip().splitlines()[-8:]:
                print(f"    {line}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _verify_remote(url: str, local_path: str) -> None:
    """Fail fast if the URL does not serve the local file's first bytes."""
    status, body, _ = http_get(url, "bytes=0-63", timeout=30)
    if status != 206:
        raise RuntimeError(f"readiness check on {url} expected 206, got {status}")
    with open(local_path, "rb") as fh:
        expected = fh.read(64)
    if body[: len(expected)] != expected:
        raise RuntimeError(
            f"{url} does not serve {local_path}: first bytes differ "
            f"(got {body[:16]!r}, expected {expected[:16]!r})"
        )


def run_harness(args: argparse.Namespace) -> int:
    file_path = os.path.abspath(args.file)
    if not os.path.isfile(file_path):
        print(f"ERROR: model file not found: {file_path}", file=sys.stderr)
        print(
            "Run: python3 -m src.bench.make_synthetic_model "
            f"--out {args.file} --layers 8 --size-mb 8",
            file=sys.stderr,
        )
        return 2

    try:
        manifest, manifest_parser = load_manifest(file_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot parse safetensors file {file_path}: {exc}", file=sys.stderr)
        return 2

    layers = manifest["layers"]
    names = list(layers)
    if not names:
        print(f"ERROR: no layers found in {file_path}", file=sys.stderr)
        return 2
    size = int(manifest["size"])
    model_name = args.model or os.path.basename(file_path)

    modes = ["baseline", "pipeline"] if args.mode == "both" else [args.mode]
    out_path = Path(args.out)
    per_layer_path = out_path.parent / "per_layer.csv"
    rss_dir = out_path.parent / "rss"
    rss_dir.mkdir(parents=True, exist_ok=True)

    server: RangeServer | None = None
    url = args.url
    print("=" * 78)
    print("Progressive layer-streaming benchmark")
    print(f"  file           : {file_path} ({size} bytes, {size / MIB:.2f} MiB)")
    print(f"  layers         : {', '.join(names)}")
    print(f"  manifest parser: {manifest_parser}")
    print(f"  mode(s)        : {', '.join(modes)}")
    print(f"  engine         : {args.engine}")
    print(f"  runs/mode      : {args.runs}")
    print(f"  bandwidth      : {args.bandwidth_mbps:g} Mbit/s (shared throttle)")
    print(f"  lookahead      : {args.lookahead}")
    print(f"  work factor    : {args.work_factor}")
    print(f"  output CSV     : {out_path}")
    print(
        "  NOTE: local RangeServer + loopback is a stand-in for MinIO/S3 + tc; "
        "production uses MinIO + tc/netem."
    )
    print("=" * 78, flush=True)

    rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    try:
        if url:
            print(f"Using external RangeServer: {url}", flush=True)
            _verify_remote(url, file_path)
        else:
            server = RangeServer(
                file_path,
                host=args.host,
                port=args.port,
                bandwidth_mbps=args.bandwidth_mbps,
                log_requests=args.server_log,
            )
            server.start()
            url = server.url
            print(f"Started in-process RangeServer: {url}", flush=True)

        if args.bandwidth_mbps > 0:
            expected = size / (args.bandwidth_mbps * 1_000_000.0 / 8.0)
        else:
            expected = 30.0
        timeout_s = args.timeout if args.timeout > 0 else max(180.0, expected * 10.0 + 120.0)

        for run_idx in range(1, args.runs + 1):
            for mode in modes:
                engine_for_note = args.engine
                cfg = {
                    "url": url,
                    "layers": layers,
                    "mode": mode,
                    "engine": args.engine,
                    "lookahead": args.lookahead,
                    "work_factor": args.work_factor,
                    "bandwidth_mbps": args.bandwidth_mbps,
                    "size": size,
                    "timeout": max(60.0, timeout_s / 2.0),
                    "rss_interval": args.rss_interval,
                    "rss_points": 400,
                    "rss_limit_mb": args.rss_limit_mb,
                }
                print(f"[run {run_idx}/{args.runs}] mode={mode} engine={args.engine} ... ", end="", flush=True)
                outcome = _run_in_subprocess(cfg, timeout_s)
                if not outcome.get("ok"):
                    failures.append(
                        {
                            "run": run_idx,
                            "mode": mode,
                            "engine": args.engine,
                            "error": outcome.get("error", "unknown failure"),
                            "traceback": outcome.get("traceback", ""),
                        }
                    )
                    print(f"FAILED: {outcome.get('error', 'unknown failure')}", flush=True)
                    continue

                result = outcome["result"]
                engine_for_note = result.get("engine", args.engine)
                ttft = result.get("ttft_s")
                first_layer = result.get("first_layer_s")
                total = result.get("total_s")
                io_wait = result.get("io_wait_s")
                compute = result.get("compute_s")
                masking = result.get("masking_ratio")
                peak = result.get("peak_rss_mb")
                nbytes = result.get("bytes_downloaded")

                note = (
                    f"engine={engine_for_note}; manifest_parser={manifest_parser}; "
                    "local RangeServer+loopback stand-in "
                    "for MinIO+tc (production: MinIO+tc)"
                )
                if engine_for_note == "fallback":
                    if args.engine == "fallback":
                        note += "; fallback engine requested via --engine fallback"
                    else:
                        note += "; src/core unavailable at run time (fallback engine)"
                if io_wait is None or masking is None:
                    note += "; core did not report io_wait/masking"

                rss_path = rss_dir / f"rss_{mode}_run{run_idx}.csv"
                trace = result.get("rss_trace") or []
                with open(rss_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["t_s", "rss_mb"])
                    for t, rss in trace:
                        writer.writerow([f"{t:.6f}", f"{rss:.4f}"])
                note += f"; rss_trace={rss_path}"

                rows.append(
                    {
                        "run": run_idx,
                        "mode": mode,
                        "ttft_s": _fmt(ttft, 6),
                        "first_layer_s": _fmt(first_layer, 6),
                        "total_s": _fmt(total, 6),
                        "io_wait_s": _fmt(io_wait, 6),
                        "compute_s": _fmt(compute, 6),
                        "masking_ratio": _fmt(masking, 6),
                        "peak_rss_mb": _fmt(peak, 3),
                        "bytes_downloaded": int(nbytes) if nbytes is not None else "TBD",
                        "bandwidth_mbps": args.bandwidth_mbps,
                        "model": model_name,
                        "notes": note,
                    }
                )
                for layer_row in result.get("per_layer") or []:
                    per_layer_rows.append(
                        {
                            "run": run_idx,
                            "mode": mode,
                            "layer": layer_row.get("layer", ""),
                            "bytes": layer_row.get("bytes", ""),
                            "fetch_s": _fmt(layer_row.get("fetch_s"), 6)
                            if layer_row.get("fetch_s") is not None
                            else "",
                            "compute_s": _fmt(layer_row.get("compute_s"), 6)
                            if layer_row.get("compute_s") is not None
                            else "",
                            "blocked_s": _fmt(layer_row.get("blocked_s"), 6)
                            if layer_row.get("blocked_s") is not None
                            else "",
                            "fetch_start_s": _fmt(layer_row.get("fetch_start_s"), 6)
                            if layer_row.get("fetch_start_s") is not None
                            else "",
                            "fetch_end_s": _fmt(layer_row.get("fetch_end_s"), 6)
                            if layer_row.get("fetch_end_s") is not None
                            else "",
                            "compute_start_s": _fmt(layer_row.get("compute_start_s"), 6)
                            if layer_row.get("compute_start_s") is not None
                            else "",
                            "compute_end_s": _fmt(layer_row.get("compute_end_s"), 6)
                            if layer_row.get("compute_end_s") is not None
                            else "",
                        }
                    )
                print(
                    f"ok  ttft={_fmt(ttft, 4)}s total={_fmt(total, 4)}s "
                    f"masking={_fmt(masking, 3)} rss={_fmt(peak, 1)}MB",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 - convert setup/run errors to a TBD row
        failures.append(
            {
                "run": "-",
                "mode": "-",
                "engine": args.engine,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if server is not None:
            server.stop()

    if not rows:
        reason = failures[0]["error"] if failures else "no runs were executed"
        _write_csv(
            out_path,
            [
                {
                    "run": "TBD",
                    "mode": "TBD",
                    "ttft_s": "TBD",
                    "first_layer_s": "TBD",
                    "total_s": "TBD",
                    "io_wait_s": "TBD",
                    "compute_s": "TBD",
                    "masking_ratio": "TBD",
                    "peak_rss_mb": "TBD",
                    "bytes_downloaded": "TBD",
                    "bandwidth_mbps": args.bandwidth_mbps,
                    "model": model_name,
                    "notes": f"TBD — pending successful run: {reason}",
                }
            ],
        )
        _write_per_layer(per_layer_path, [])
        _print_failures(failures)
        print(f"\nTBD — pending successful run (no successful benchmark run).")
        print(f"Reason: {reason}")
        print(f"CSV written with TBD note: {out_path}")
        print(
            "INTEGRATION: ensure src/core exists (SafetensorsManifest, AsyncPrefetcher, "
            "MemoryBuffer, PipelineDriver) and run:"
        )
        print(
            f"  python3 -m src.bench.BenchmarkHarness --file {args.file} "
            f"--bandwidth-mbps {args.bandwidth_mbps:g} --runs {args.runs} "
            f"--lookahead {args.lookahead} --work-factor {args.work_factor} "
            f"--mode {args.mode} --out {args.out}"
        )
        return 2

    _write_csv(out_path, rows)
    _write_per_layer(per_layer_path, per_layer_rows)
    print(f"\nCSV written: {out_path}")
    print(f"Per-layer detail: {per_layer_path}")
    print(f"RSS trajectories: {rss_dir}/")
    _print_stats(rows)
    _print_failures(failures)

    if failures:
        print(
            f"\nWARNING: {len(failures)} run(s) failed; successful rows are real, "
            "failed runs are excluded (not replaced with fabricated numbers)."
        )
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark monolithic vs progressive layer-streaming "
        "for serverless LLM invocations (local RangeServer stand-in)."
    )
    p.add_argument("--file", default="models/synthetic_8layer.safetensors",
                   help="safetensors model to serve/benchmark (8-layer synthetic default; "
                        "models/synthetic_2layer.safetensors is the minimal artifact)")
    p.add_argument("--url", default=None,
                   help="use an already-running RangeServer URL instead of starting one in-process")
    p.add_argument("--host", default="127.0.0.1", help="bind host for in-process server")
    p.add_argument("--port", type=int, default=0,
                   help="bind port for in-process server (0 = ephemeral)")
    p.add_argument("--bandwidth-mbps", type=float, default=8.0,
                   help="simulated shared link rate in Mbit/s (0 = unlimited)")
    p.add_argument("--runs", type=int, default=3, help="repetitions per mode")
    p.add_argument("--lookahead", type=int, default=2, help="prefetch lookahead depth")
    p.add_argument("--work-factor", type=float, default=1.0,
                   help="synthetic per-layer compute multiplier")
    p.add_argument("--mode", choices=["both", "baseline", "pipeline"], default="both")
    p.add_argument("--out", default="results/benchmark.csv", help="output CSV path")
    p.add_argument("--engine", choices=["auto", "core", "fallback"], default="auto",
                   help="auto: use src/core if importable, else fallback")
    p.add_argument("--model", default=None, help="model name recorded in the CSV")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="per-run timeout seconds (0 = derive from size/bandwidth)")
    p.add_argument("--rss-interval", type=float, default=0.02,
                   help="RSS sampling interval seconds")
    p.add_argument("--rss-limit-mb", type=float, default=1024.0,
                   help="rss_limit_mb passed to MemoryBuffer when available")
    p.add_argument("--server-log", action="store_true",
                   help="enable per-request RangeServer logging (in-process server)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.runs < 1:
        print("ERROR: --runs must be >= 1", file=sys.stderr)
        return 2
    if args.lookahead < 1:
        print("ERROR: --lookahead must be >= 1", file=sys.stderr)
        return 2
    if args.work_factor < 0:
        print("ERROR: --work-factor must be >= 0", file=sys.stderr)
        return 2
    try:
        return run_harness(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI must fail honestly
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
