#!/usr/bin/env python3
"""End-to-end benchmark / verification harness for progressive layer streaming.

What this script does
---------------------
1. Starts ``src.storage.RangeServer.RangeServer`` on ``127.0.0.1:8001`` (the
   server behind ``python3 -m src.storage.RangeServer``) with a configurable
   shared-link throttle (default 1000 Mbit/s), then invokes
   ``src/bench/BenchmarkHarness.py`` in ``--mode both`` (baseline vs pipeline)
   against that live server.  The BenchmarkHarness rows use its *synthetic*
   compute engine; they are labelled as such and are not presented as real GEMM.
2. Runs the real TinyLlama-1.1B checkpoint through the production pipeline
   (``RealModelComputeEngine`` + ``RangeServer`` + ``AsyncPrefetcher`` +
   ``PipelineDriver``), 22 transformer blocks in numeric order, one token
   (``seq=1``).  It then runs the download-all-then-compute baseline with the
   same real engine so the two strategies are comparable.
3. Verifies the real run: exactly 22 layers, finite checksums that rise overall,
   first-layer latency (streaming fill), measured compute time and measured
   I/O wait.  A first-layer time above the ~2 s target is reported as a warning
   (the link is the local loopback stand-in, not ``tc``).
4. Prints a table containing bandwidth, first_layer_s, total_s, compute_s,
   io_wait, masking_ratio and peak_rss_mb for every measured run.
5. When CUDA is available it repeats the real pipeline on CPU and GPU and prints
   a per-layer comparison.  At ``seq=1`` the GPU is often slower than CPU
   OpenBLAS because each tiny GEMM pays kernel-launch and host->device costs;
   the measured result is reported as-is.
6. Unless ``--skip-demo`` is passed, starts ``web/demo_server.py``, checks that
   ``http://127.0.0.1:8000/demo.html`` is HTTP 200, and checks that
   ``/results.json`` contains 22 rows with finite, rising checksums plus the
   final summary fields.

Honesty notes
-------------
* The local RangeServer + HTTP loopback is a stand-in for MinIO/S3 + ``tc``.
  Its measured throughput is printed; if it is below the nominal rate,
  ``masking_ratio = clamp(1 - io_wait / nominal_io, 0, 1)`` clamps to 0.
* ``masking_ratio`` here uses the same definition as BenchmarkHarness:
  ``1 - io_wait / nominal_io`` with ``nominal_io = bytes / (bandwidth / 8)``.
  It is not presented as a production masking number.
* Only measured numbers are printed.  No synthetic compute result is labelled
  as real GEMM; the real runs use the real BF16 tensors.

Run from the repository root::

    python3 examples/end_to_end.py
    python3 examples/end_to_end.py --bandwidth-mbps 1000 --skip-demo
    python3 examples/end_to_end.py --device cpu --skip-harness
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.bench.BenchmarkHarness import RSSSampler, http_get  # noqa: E402
from src.core.AsyncPrefetcher import AsyncPrefetcher  # noqa: E402
from src.core.MemoryBuffer import MemoryBuffer  # noqa: E402
from src.core.PipelineDriver import PipelineDriver  # noqa: E402
from src.core.RealModelComputeEngine import RealModelComputeEngine  # noqa: E402
from src.core.SafetensorsManifest import SafetensorsManifest  # noqa: E402
from src.storage.RangeServer import RangeServer  # noqa: E402

MODEL = ROOT / "models" / "tinyllama.safetensors"
HARNESS_CSV = ROOT / "results" / "end_to_end_harness.csv"
REAL_CSV = ROOT / "results" / "end_to_end.csv"
DEMO_LOG = ROOT / "results" / "demo_server.log"
EXPECTED_LAYERS = 22
DEFAULT_TTFT_LIMIT_S = 2.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "TBD"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "TBD"
    return f"{number:.{digits}f}"


def _forward_order(keys: Iterable[str]) -> list[str]:
    """Return transformer blocks in numeric forward order (0, 1, ..., 21)."""

    def index(name: str) -> int:
        tail = name.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else 1 << 30

    return sorted(keys, key=lambda name: (index(name), name))


def _masking_ratio(io_wait_s: float | None, nbytes: int, bandwidth_mbps: float) -> float:
    """BenchmarkHarness definition: 1 - io_wait / nominal_io, clamped to [0, 1]."""
    if not bandwidth_mbps or bandwidth_mbps <= 0 or nbytes <= 0 or io_wait_s is None:
        return 0.0
    nominal_io = nbytes / (bandwidth_mbps * 1_000_000.0 / 8.0)
    return _clamp01(1.0 - float(io_wait_s) / nominal_io)


def _layer_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [
        float(row["compute_s"])
        for row in rows
        if row.get("compute_s") is not None
    ]
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


# ---------------------------------------------------------------------------
# RangeServer lifecycle
# ---------------------------------------------------------------------------
def _start_server(model_path: Path, host: str, port: int, bandwidth_mbps: float) -> RangeServer:
    server = RangeServer(
        str(model_path),
        host=host,
        port=port,
        bandwidth_mbps=bandwidth_mbps,
        log_requests=False,
    )
    server.start()
    return server


def _server_for(
    model_path: Path, host: str, port: int, bandwidth_mbps: float
) -> RangeServer:
    try:
        return _start_server(model_path, host, port, bandwidth_mbps)
    except OSError as exc:
        raise RuntimeError(
            f"cannot start RangeServer on {host}:{port} ({exc}); "
            "stop the process using that port or pass --port"
        ) from exc


# ---------------------------------------------------------------------------
# 1a. BenchmarkHarness (synthetic compute engine, baseline vs pipeline)
# ---------------------------------------------------------------------------
def _run_benchmark_harness(
    model_path: Path,
    host: str,
    port: int,
    bandwidth_mbps: float,
    runs: int,
    work_factor: float,
    lookahead: int,
    timeout_s: float,
) -> tuple[int, list[dict[str, str]]]:
    print("\n" + "=" * 100)
    print("STEP 1a: BenchmarkHarness baseline vs pipeline (synthetic compute engine)")
    print("=" * 100, flush=True)

    server = _server_for(model_path, host, port, bandwidth_mbps)
    try:
        url = server.url
        print(f"RangeServer : {url}")
        print(f"file        : {model_path} ({server.size:,} bytes)")
        print(f"throttle    : {bandwidth_mbps:g} Mbit/s (shared across concurrent ranges)")
        print(f"engine      : synthetic (BenchmarkHarness.SyntheticComputeEngine), "
              f"work_factor={work_factor:g}")
        print(f"runs/mode   : {runs}   lookahead={lookahead}", flush=True)

        cmd = [
            sys.executable,
            "-m",
            "src.bench.BenchmarkHarness",
            "--file",
            str(model_path),
            "--url",
            url,
            "--mode",
            "both",
            "--runs",
            str(runs),
            "--bandwidth-mbps",
            str(bandwidth_mbps),
            "--work-factor",
            str(work_factor),
            "--lookahead",
            str(lookahead),
            "--engine",
            "core",
            "--out",
            str(HARNESS_CSV),
        ]
        print("$ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout_s)
        print(f"BenchmarkHarness exit code: {proc.returncode}", flush=True)
    finally:
        server.stop()

    rows: list[dict[str, str]] = []
    if HARNESS_CSV.is_file():
        with HARNESS_CSV.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("mode") in ("baseline", "pipeline")]
    if rows:
        print("\nBenchmarkHarness measured summary (synthetic compute; real checkpoint bytes)")
        header = (
            f"{'mode':<10} {'bandwidth':>10} {'first_layer_s':>14} {'total_s':>10} "
            f"{'compute_s':>10} {'io_wait_s':>10} {'masking':>8} {'peak_rss_mb':>12}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row.get('mode', ''):<10} {row.get('bandwidth_mbps', ''):>10} "
                f"{_fmt(row.get('first_layer_s')):>14} {_fmt(row.get('total_s')):>10} "
                f"{_fmt(row.get('compute_s')):>10} {_fmt(row.get('io_wait_s')):>10} "
                f"{_fmt(row.get('masking_ratio'), 3):>8} {_fmt(row.get('peak_rss_mb'), 1):>12}"
            )
    else:
        print("\nBenchmarkHarness produced no parseable rows.", flush=True)
    return proc.returncode, rows


# ---------------------------------------------------------------------------
# 1b/1e. Real-model pipeline and real-model baseline
# ---------------------------------------------------------------------------
def _build_layers(manifest: SafetensorsManifest, names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "start": int(manifest.layers[name].start),
            "end": int(manifest.layers[name].end),
            "dtype": manifest.layers[name].dtype,
        }
        for name in names
    }


async def _pipeline_async(
    engine: RealModelComputeEngine,
    url: str,
    layers: dict[str, dict[str, Any]],
    names: list[str],
    bandwidth_mbps: float,
    lookahead: int,
    rss_limit_mb: float,
) -> dict[str, Any]:
    buffer = MemoryBuffer(
        max_depth=max(2, lookahead + 1),
        rss_limit_mb=rss_limit_mb,
    )
    prefetcher = AsyncPrefetcher(
        url,
        layers,
        buffer=buffer,
        lookahead=lookahead,
        timeout=600.0,
        retries=2,
    )
    engine._init_hidden()
    sampler = RSSSampler(interval=0.05)
    sampler.start()
    try:
        await prefetcher.start()
        driver = PipelineDriver(
            prefetcher,
            compute_engine=engine,
            buffer=buffer,
            sample_rss=False,
        )
        metrics = await driver.run(names)
    finally:
        await prefetcher.stop()
        sampler.stop()

    rows = metrics.get("per_layer") or []
    nbytes = int(metrics.get("bytes_downloaded") or 0)
    io_wait = metrics.get("io_wait_s")
    metrics["peak_rss_mb"] = sampler.peak()
    metrics["nominal_io_s"] = (
        nbytes / (bandwidth_mbps * 1_000_000.0 / 8.0)
        if bandwidth_mbps > 0 and nbytes > 0
        else None
    )
    metrics["masking_ratio"] = _masking_ratio(io_wait, nbytes, bandwidth_mbps)
    metrics["layer_stats"] = _layer_stats(rows)
    metrics["prefetcher_measured_mbps"] = (
        (nbytes * 8.0 / 1_000_000.0) / metrics["total_s"]
        if metrics.get("total_s")
        else None
    )
    return metrics


def _run_real_pipeline(
    engine: RealModelComputeEngine,
    url: str,
    layers: dict[str, dict[str, Any]],
    names: list[str],
    bandwidth_mbps: float,
    lookahead: int,
    rss_limit_mb: float,
) -> dict[str, Any]:
    return asyncio.run(
        _pipeline_async(
            engine,
            url,
            layers,
            names,
            bandwidth_mbps,
            lookahead,
            rss_limit_mb,
        )
    )


def _run_real_baseline(
    engine: RealModelComputeEngine,
    url: str,
    layers: dict[str, dict[str, Any]],
    names: list[str],
    bandwidth_mbps: float,
) -> dict[str, Any]:
    """Download the entire object, then run every real layer from memory."""
    engine._init_hidden()
    sampler = RSSSampler(interval=0.05)
    sampler.start()
    t0 = time.perf_counter()
    try:
        status, body, headers = http_get(url, timeout=600.0)
        download_s = time.perf_counter() - t0
        if status != 200:
            raise RuntimeError(
                f"baseline full GET expected HTTP 200, got {status} "
                f"(Content-Range={headers.get('content-range')!r})"
            )
        expected_size = int(layers[names[-1]]["end"])
        if len(body) < expected_size:
            raise RuntimeError(
                f"baseline downloaded {len(body)} bytes, expected at least {expected_size}"
            )

        per_layer: list[dict[str, Any]] = []
        compute_total = 0.0
        first_layer_s: float | None = None
        for index, name in enumerate(names):
            start = int(layers[name]["start"])
            end = int(layers[name]["end"])
            payload = body[start:end]
            c_start = time.perf_counter()
            result = engine.run_layer(name, payload)
            c_end = time.perf_counter()
            compute_total += c_end - c_start
            if index == 0:
                first_layer_s = c_end - t0
            per_layer.append(
                {
                    "layer": name,
                    "bytes": len(payload),
                    "fetch_s": None,
                    "download_s": None,
                    "blocked_s": None,
                    "compute_s": c_end - c_start,
                    "checksum": float(getattr(result, "checksum", float("nan"))),
                }
            )
        total_s = time.perf_counter() - t0
    finally:
        sampler.stop()

    nbytes = len(body)
    return {
        "ttft_s": total_s,
        "first_layer_s": first_layer_s,
        "total_s": total_s,
        "io_wait_s": download_s,
        "compute_s": compute_total,
        "masking_ratio": _masking_ratio(download_s, nbytes, bandwidth_mbps),
        "peak_rss_mb": sampler.peak(),
        "bytes_downloaded": nbytes,
        "nominal_io_s": (
            nbytes / (bandwidth_mbps * 1_000_000.0 / 8.0)
            if bandwidth_mbps > 0
            else None
        ),
        "measured_loopback_mbps": (
            nbytes * 8.0 / 1_000_000.0 / download_s if download_s > 0 else None
        ),
        "per_layer": per_layer,
        "layer_stats": _layer_stats(per_layer),
    }


def _verify_real_run(
    label: str,
    result: dict[str, Any],
    ttft_limit_s: float,
) -> bool:
    rows = result.get("per_layer") or []
    checks = [float(row.get("checksum")) for row in rows if row.get("checksum") is not None]
    failures: list[str] = []
    warnings: list[str] = []

    if len(rows) != EXPECTED_LAYERS:
        failures.append(f"expected {EXPECTED_LAYERS} layers, got {len(rows)}")
    if len(checks) != len(rows):
        failures.append("some layers have no checksum")
    if checks and not all(math.isfinite(value) for value in checks):
        failures.append("non-finite checksum detected")
    if len(checks) >= 2:
        increasing_steps = sum(1 for a, b in zip(checks, checks[1:]) if b > a)
        if checks[-1] <= checks[0]:
            failures.append(
                f"checksum did not rise overall ({checks[0]:.6f} -> {checks[-1]:.6f})"
            )
        elif increasing_steps < len(checks) - 1:
            warnings.append(
                f"checksums rise overall but {len(checks) - 1 - increasing_steps} "
                "step(s) are flat/down"
            )
    first_layer = result.get("first_layer_s")
    if first_layer is None:
        failures.append("first_layer_s was not measured")
    elif float(first_layer) > ttft_limit_s:
        warnings.append(
            f"first_layer_s {float(first_layer):.4f}s exceeds ~{ttft_limit_s:.1f}s target"
        )
    if result.get("compute_s") is None:
        failures.append("compute_s was not measured")
    if result.get("io_wait_s") is None:
        failures.append("io_wait_s was not measured")
    if result.get("total_s") is None or float(result.get("total_s") or 0) <= 0:
        failures.append("total_s was not measured")

    increasing_steps = (
        sum(1 for a, b in zip(checks, checks[1:]) if b > a)
        if len(checks) >= 2
        else 0
    )
    print(
        f"  verify {label}: layers={len(rows)}/{EXPECTED_LAYERS} "
        f"checksums={_fmt(checks[0] if checks else None, 6)}->{_fmt(checks[-1] if checks else None, 6)} "
        f"rising_steps={increasing_steps}/{max(0, len(checks) - 1)} "
        f"finite={bool(checks) and all(math.isfinite(v) for v in checks)} "
        f"first_layer={_fmt(first_layer)}s compute={_fmt(result.get('compute_s'))}s "
        f"io_wait={_fmt(result.get('io_wait_s'))}s"
    )
    for warning in warnings:
        print(f"  WARNING {label}: {warning}")
    for failure in failures:
        print(f"  FAIL    {label}: {failure}")
    if not failures:
        print(f"  PASS    {label}: required measurements present")
    return not failures


def _print_real_table(results: list[dict[str, Any]], bandwidth_mbps: float) -> None:
    print("\nReal-model measured table (TinyLlama-1.1B, 22 layers, seq=1)")
    header = (
        f"{'mode':<10} {'device':<7} {'bandwidth':>10} {'first_layer_s':>14} "
        f"{'total_s':>10} {'compute_s':>10} {'io_wait_s':>10} {'masking':>8} "
        f"{'peak_rss_mb':>12} {'mean_layer_s':>13}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        stats = result.get("layer_stats") or {}
        print(
            f"{result['mode']:<10} {result['device']:<7} {bandwidth_mbps:>10.0f} "
            f"{_fmt(result.get('first_layer_s')):>14} {_fmt(result.get('total_s')):>10} "
            f"{_fmt(result.get('compute_s')):>10} {_fmt(result.get('io_wait_s')):>10} "
            f"{_fmt(result.get('masking_ratio'), 3):>8} {_fmt(result.get('peak_rss_mb'), 1):>12} "
            f"{_fmt(stats.get('mean')):>13}"
        )


def _print_device_comparison(results: list[dict[str, Any]]) -> None:
    pipelines = {
        result["device"]: result
        for result in results
        if result["mode"] == "pipeline"
    }
    if "cpu" not in pipelines or "cuda" not in pipelines:
        return
    cpu = pipelines["cpu"]
    cuda = pipelines["cuda"]
    print("\nCPU vs GPU (CUDA available=True): real pipeline, seq=1 single token")
    print(
        f"  first_layer_s : CPU {_fmt(cpu.get('first_layer_s'))} s | "
        f"CUDA {_fmt(cuda.get('first_layer_s'))} s"
    )
    for device, result in (("CPU", cpu), ("CUDA", cuda)):
        stats = result.get("layer_stats") or {}
        print(
            f"  {device:<4} per-layer compute_s: mean {_fmt(stats.get('mean'))} | "
            f"median {_fmt(stats.get('median'))} | min {_fmt(stats.get('min'))} | "
            f"max {_fmt(stats.get('max'))} | total {_fmt(result.get('compute_s'))}"
        )
    print(
        "  note: at seq=1 each GEMM is tiny; CUDA kernel-launch and host->device "
        "copy overhead can exceed the CPU OpenBLAS matvec time. Reported as measured."
    )
    print("\nPer-layer compute_s (real pipeline)")
    print(f"  {'layer':<20} {'CPU':>10} {'CUDA':>10}")
    cpu_rows = {row["layer"]: row for row in cpu.get("per_layer") or []}
    cuda_rows = {row["layer"]: row for row in cuda.get("per_layer") or []}
    for name in _forward_order(cpu_rows.keys()):
        print(
            f"  {name:<20} {_fmt(cpu_rows[name].get('compute_s')):>10} "
            f"{_fmt(cuda_rows.get(name, {}).get('compute_s')):>10}"
        )


def _write_real_csv(results: list[dict[str, Any]]) -> None:
    REAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "device",
        "mode",
        "layer",
        "bytes",
        "fetch_s",
        "compute_s",
        "blocked_s",
        "checksum",
        "metric",
        "value",
    ]
    with REAL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            for row in result.get("per_layer") or []:
                writer.writerow(
                    {
                        "device": result["device"],
                        "mode": result["mode"],
                        "layer": row.get("layer"),
                        "bytes": row.get("bytes"),
                        "fetch_s": row.get("fetch_s"),
                        "compute_s": row.get("compute_s"),
                        "blocked_s": row.get("blocked_s"),
                        "checksum": row.get("checksum"),
                    }
                )
            for metric in (
                "first_layer_s",
                "ttft_s",
                "total_s",
                "compute_s",
                "io_wait_s",
                "masking_ratio",
                "peak_rss_mb",
                "bytes_downloaded",
                "nominal_io_s",
                "measured_loopback_mbps",
                "prefetcher_measured_mbps",
            ):
                writer.writerow(
                    {
                        "device": result["device"],
                        "mode": result["mode"],
                        "metric": metric,
                        "value": result.get(metric),
                    }
                )
            writer.writerow({})


def _run_real_section(
    args: argparse.Namespace,
    host: str,
    port: int,
) -> tuple[int, list[dict[str, Any]]]:
    print("\n" + "=" * 100)
    print("STEP 1b/1e: real TinyLlama pipeline vs download-all-then-compute baseline")
    print("=" * 100, flush=True)

    manifest = SafetensorsManifest(str(MODEL))
    real_names = _forward_order(
        name for name in manifest.layers if name.startswith("model.layers.")
    )
    if len(real_names) != EXPECTED_LAYERS:
        raise RuntimeError(
            f"expected {EXPECTED_LAYERS} transformer blocks in {MODEL}, "
            f"found {len(real_names)}"
        )
    layers = _build_layers(manifest, real_names)

    print(f"model       : {MODEL} ({manifest.file_size:,} bytes)")
    print(f"layers      : {len(real_names)} ({real_names[0]} .. {real_names[-1]})")
    print(f"dtype       : {manifest.layers[real_names[0]].dtype}; seq=1 single-token")
    print(f"torch       : {torch.__version__}")
    print(f"CUDA        : available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"lookahead   : {args.lookahead}")
    print(f"throttle    : {args.bandwidth_mbps:g} Mbit/s (shared)", flush=True)

    if args.device == "auto":
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    elif args.device == "both":
        if not torch.cuda.is_available():
            raise RuntimeError("--device both requires torch.cuda.is_available()")
        devices = ["cpu", "cuda"]
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        devices = ["cuda"]
    else:
        devices = ["cpu"]

    server = _server_for(MODEL, host, port, args.bandwidth_mbps)
    results: list[dict[str, Any]] = []
    verification_failures = 0
    try:
        url = server.url
        print(f"RangeServer : {url}", flush=True)
        for device in devices:
            print(f"\n--- device={device} ---", flush=True)
            try:
                engine = RealModelComputeEngine(str(MODEL), device=torch.device(device))
            except Exception as exc:  # noqa: BLE001 - report and continue other devices
                verification_failures += 1
                print(f"FAIL cannot construct real engine on {device}: {type(exc).__name__}: {exc}")
                continue

            print(f"[pipeline] {device}: AsyncPrefetcher + PipelineDriver + RealModelComputeEngine ...",
                  end="", flush=True)
            try:
                pipeline = _run_real_pipeline(
                    engine,
                    url,
                    layers,
                    real_names,
                    args.bandwidth_mbps,
                    args.lookahead,
                    args.rss_limit_mb,
                )
            except Exception as exc:  # noqa: BLE001
                verification_failures += 1
                print(f" FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            pipeline["device"] = device
            pipeline["mode"] = "pipeline"
            results.append(pipeline)
            print(
                f" ok first_layer={_fmt(pipeline.get('first_layer_s'))}s "
                f"total={_fmt(pipeline.get('total_s'))}s "
                f"compute={_fmt(pipeline.get('compute_s'))}s "
                f"io_wait={_fmt(pipeline.get('io_wait_s'))}s "
                f"masking={_fmt(pipeline.get('masking_ratio'), 3)} "
                f"rss={_fmt(pipeline.get('peak_rss_mb'), 1)}MB",
                flush=True,
            )
            if not _verify_real_run(f"{device}/pipeline", pipeline, args.ttft_limit_s):
                verification_failures += 1

            if not args.skip_baseline:
                print(f"[baseline] {device}: full GET then sequential real layers ...",
                      end="", flush=True)
                try:
                    baseline = _run_real_baseline(
                        engine, url, layers, real_names, args.bandwidth_mbps
                    )
                except Exception as exc:  # noqa: BLE001
                    verification_failures += 1
                    print(f" FAILED: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                else:
                    baseline["device"] = device
                    baseline["mode"] = "baseline"
                    results.append(baseline)
                    print(
                        f" ok first_layer={_fmt(baseline.get('first_layer_s'))}s "
                        f"total={_fmt(baseline.get('total_s'))}s "
                        f"compute={_fmt(baseline.get('compute_s'))}s "
                        f"io_wait={_fmt(baseline.get('io_wait_s'))}s "
                        f"masking={_fmt(baseline.get('masking_ratio'), 3)} "
                        f"rss={_fmt(baseline.get('peak_rss_mb'), 1)}MB "
                        f"loopback={_fmt(baseline.get('measured_loopback_mbps'), 1)}Mbit/s",
                        flush=True,
                    )
                    if not _verify_real_run(f"{device}/baseline", baseline, args.ttft_limit_s):
                        verification_failures += 1

            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    finally:
        server.stop()

    if not results:
        raise RuntimeError("no real-model runs completed")

    _print_real_table(results, args.bandwidth_mbps)
    _print_device_comparison(results)

    baseline_rows = [r for r in results if r["mode"] == "baseline"]
    if baseline_rows:
        loopback = [
            float(r["measured_loopback_mbps"])
            for r in baseline_rows
            if r.get("measured_loopback_mbps")
        ]
        if loopback:
            measured = statistics.fmean(loopback)
            print(
                f"\nMeasured loopback throughput (baseline full GET): {measured:.1f} Mbit/s "
                f"vs {args.bandwidth_mbps:g} Mbit/s nominal."
            )
            if measured < args.bandwidth_mbps:
                print(
                    "  Loopback is slower than nominal (per-chunk pacing overhead in "
                    "RangeServer), so io_wait >= nominal_io and masking_ratio clamps to 0."
                )

    _write_real_csv(results)
    print(f"\nWrote real-model per-layer + summary rows: {REAL_CSV}")
    return verification_failures, results


# ---------------------------------------------------------------------------
# 2. Live demo verification
# ---------------------------------------------------------------------------
def _http_status(url: str, timeout: float = 10.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        response.read()
        return int(response.status)


def _http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _verify_live_demo(timeout_s: float = 240.0) -> bool:
    print("\n" + "=" * 100)
    print("STEP 2: live demo server (web/demo_server.py)")
    print("=" * 100, flush=True)

    log_handle = DEMO_LOG.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "web/demo_server.py"],
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    failures: list[str] = []
    try:
        demo_url = "http://127.0.0.1:8000/demo.html"
        results_url = "http://127.0.0.1:8000/results.json"
        deadline = time.monotonic() + timeout_s
        demo_status: int | None = None
        payload: dict[str, Any] | None = None
        last_error = ""

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                failures.append(f"demo_server exited early with rc={proc.returncode}")
                break
            try:
                if demo_status is None:
                    demo_status = _http_status(demo_url)
                snapshot = _http_json(results_url)
                payload = snapshot
                rows = snapshot.get("rows") or []
                final = snapshot.get("final")
                status = snapshot.get("status")
                if status == "done" and final is not None and len(rows) >= EXPECTED_LAYERS:
                    break
                if isinstance(status, str) and status.startswith("error"):
                    failures.append(f"demo run status={status!r}")
                    break
            except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0)
        else:
            failures.append(f"timed out after {timeout_s:.0f}s waiting for demo results")

        if demo_status != 200:
            failures.append(f"GET /demo.html returned {demo_status!r}, expected 200")
        else:
            print(f"  GET /demo.html -> HTTP {demo_status}")

        if payload is None:
            failures.append(f"could not read /results.json ({last_error or 'no response'})")
        else:
            rows = payload.get("rows") or []
            final = payload.get("final")
            checks = [float(row.get("checksum")) for row in rows if row.get("checksum") is not None]
            print(f"  /results.json status={payload.get('status')!r} rows={len(rows)}")
            if len(rows) != EXPECTED_LAYERS:
                failures.append(f"/results.json rows={len(rows)}, expected {EXPECTED_LAYERS}")
            if len(checks) != len(rows):
                failures.append("/results.json has rows without checksum")
            if checks and not all(math.isfinite(value) for value in checks):
                failures.append("/results.json has non-finite checksum")
            if len(checks) >= 2:
                rising_steps = sum(1 for a, b in zip(checks, checks[1:]) if b > a)
                if checks[-1] <= checks[0]:
                    failures.append(
                        f"/results.json checksums do not rise overall "
                        f"({checks[0]:.6f} -> {checks[-1]:.6f})"
                    )
                print(
                    f"  checksums {checks[0]:.6f} -> {checks[-1]:.6f}; "
                    f"rising_steps={rising_steps}/{len(checks) - 1}"
                )
            required = (
                "first_layer_s",
                "total_s",
                "compute_s",
                "io_wait_s",
                "masking_ratio",
                "peak_rss_mb",
                "runtime",
            )
            if not isinstance(final, dict):
                failures.append("/results.json final summary is missing")
            else:
                missing = [key for key in required if key not in final]
                if missing:
                    failures.append(f"/results.json final missing keys: {missing}")
                else:
                    print("  final summary:")
                    for key in required:
                        print(f"    {key}: {final.get(key)}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log_handle.close()
        if DEMO_LOG.is_file():
            tail = DEMO_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            if tail:
                print("  demo_server log (tail):")
                for line in tail[-8:]:
                    print(f"    {line}")

    for failure in failures:
        print(f"  FAIL demo: {failure}")
    if not failures:
        print("  PASS demo: /demo.html served and /results.json has 22 rising finite rows + final summary")
    return not failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end BenchmarkHarness + real TinyLlama streaming pipeline, "
            "baseline comparison, optional CPU/CUDA comparison and live-demo check."
        )
    )
    parser.add_argument("--bandwidth-mbps", type=float, default=1000.0,
                        help="shared link rate for RangeServer (default 1000 Mbit/s)")
    parser.add_argument("--host", default="127.0.0.1", help="RangeServer bind host")
    parser.add_argument("--port", type=int, default=8001, help="RangeServer port (default 8001)")
    parser.add_argument("--lookahead", type=int, default=1,
                        help="real-pipeline AsyncPrefetcher lookahead (default 1)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "both"], default="auto",
                        help="real-model device(s); auto = CPU+CUDA when CUDA is available")
    parser.add_argument("--ttft-limit-s", type=float, default=DEFAULT_TTFT_LIMIT_S,
                        help="first-layer target in seconds, reported as warning above it")
    parser.add_argument("--rss-limit-mb", type=float, default=8192.0,
                        help="MemoryBuffer RSS clamp for the real pipeline")
    parser.add_argument("--harness-runs", type=int, default=1,
                        help="BenchmarkHarness repetitions per mode (default 1)")
    parser.add_argument("--harness-work-factor", type=float, default=0.01,
                        help="BenchmarkHarness synthetic work factor (default 0.01)")
    parser.add_argument("--harness-timeout", type=float, default=900.0,
                        help="timeout for the BenchmarkHarness subprocess")
    parser.add_argument("--skip-harness", action="store_true",
                        help="skip the BenchmarkHarness baseline-vs-pipeline section")
    parser.add_argument("--skip-real", action="store_true",
                        help="skip the real-model pipeline/baseline section")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="skip the real-model download-all-then-compute baseline")
    parser.add_argument("--skip-demo", action="store_true",
                        help="skip starting and checking web/demo_server.py")
    parser.add_argument("--demo-timeout", type=float, default=240.0,
                        help="seconds to wait for the live demo to finish")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.bandwidth_mbps < 0:
        print("ERROR: --bandwidth-mbps must be >= 0", file=sys.stderr)
        return 2
    if args.lookahead < 0:
        print("ERROR: --lookahead must be >= 0", file=sys.stderr)
        return 2
    if not MODEL.is_file():
        print(f"ERROR: model not found: {MODEL}", file=sys.stderr)
        return 2

    print("=" * 100)
    print("Progressive layer streaming: end-to-end harness")
    print("=" * 100)
    print(f"repository   : {ROOT}")
    print(f"model        : {MODEL} ({MODEL.stat().st_size:,} bytes)")
    print(f"bandwidth    : {args.bandwidth_mbps:g} Mbit/s nominal (RangeServer shared throttle)")
    print(f"torch        : {torch.__version__}")
    print(f"CUDA         : torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device  : {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA device  : none (CPU-only run)")
    print(
        "NOTE: local RangeServer + HTTP loopback is a stand-in for MinIO/S3 + tc; "
        "the measured loopback rate is printed below."
    )
    print("=" * 100, flush=True)

    failures = 0
    harness_rc = 0

    if not args.skip_harness:
        try:
            harness_rc, _rows = _run_benchmark_harness(
                MODEL,
                args.host,
                args.port,
                args.bandwidth_mbps,
                args.harness_runs,
                args.harness_work_factor,
                args.lookahead,
                args.harness_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL BenchmarkHarness section: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        if harness_rc != 0:
            failures += 1
            print(f"FAIL BenchmarkHarness exited {harness_rc}")

    if not args.skip_real:
        try:
            real_failures, _results = _run_real_section(args, args.host, args.port)
            failures += real_failures
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL real-model section: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if not args.skip_demo:
        try:
            if not _verify_live_demo(args.demo_timeout):
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL live-demo verification: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 100)
    if failures:
        print(f"RESULT: {failures} failure(s). See FAIL lines above.")
        print("=" * 100)
        return 1
    print("RESULT: all requested checks passed.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
