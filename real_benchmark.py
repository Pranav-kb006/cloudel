"""Clean real-model benchmark: real-engine pipeline through the driving path.

Wires RealModelComputeEngine through the FallbackDriver/FallbackPrefetcher
(the same per-layer contract as src.core.PipelineDriver) over a throttled
RangeServer, runs:

  1. progressive (streaming) pass  — layer fetch overlaps compute, and
  2. baseline (full pre-download)  — download the whole file, then run layers,

both with the REAL TinyLlama-1.1B-Chat weights at seq=1, and writes
results/real_model_end_to_end.csv (per-layer rows + summary rows).

Run:  python3 real_benchmark.py
"""

from __future__ import annotations

import asyncio
import csv
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.bench.BenchmarkHarness import (  # noqa: E402
    FallbackDriver,
    FallbackPrefetcher,
    RSSSampler,
    RangeServer,
    http_get,
)
from src.core.RealModelComputeEngine import RealModelComputeEngine  # noqa: E402

MODEL = os.path.join(ROOT, "models", "tinyllama.safetensors")
OUT_CSV = os.path.join(ROOT, "results", "real_model_end_to_end.csv")
REAL = RealModelComputeEngine(MODEL)  # auto-selects cuda when available

# Per-layer keys from BenchmarkHarness.py (FallbackPrefetcher/FallbackDriver).
PER_LAYER_FIELDS = [
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


def _forward_order(keys: list[str]) -> list[str]:
    """Numeric layer order (0, 1, ..., 21), not file/lexicographic order.

    The checkpoint stores tensor blocks in lexicographic key order
    (0, 1, 10, 11, ..., 2, 20, ...), but a transformer forward pass must apply
    the blocks in numeric order, so the hidden state flows 0 -> 21.
    """

    def index(name: str) -> int:
        tail = name.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else 1 << 30

    return sorted(keys, key=lambda n: (index(n), n))


class _RecordingEngine:
    """Delegates to the real engine and records each ComputeResult."""

    def __init__(self, engine: RealModelComputeEngine):
        self.engine = engine
        self.results: dict[str, object] = {}

    def run_layer(self, name: str, payload: bytes, **kwargs):
        result = self.engine.run_layer(name, payload, **kwargs)
        self.results[name] = result
        return result


def _reset_engine() -> None:
    """Fresh hidden state so each run is an independent forward pass."""
    REAL._init_hidden()


def _run_progressive(url: str, layers: dict[str, dict], names: list[str],
                     bandwidth_mbps: float) -> dict:
    pref = FallbackPrefetcher(url, layers, lookahead=2, timeout=600)
    recorder = _RecordingEngine(REAL)
    sampler = RSSSampler(interval=0.05)
    sampler.start()
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(pref.start())
        try:
            _reset_engine()
            driver = FallbackDriver(pref, recorder, names, bandwidth_mbps)
            real = loop.run_until_complete(driver.run())
        finally:
            try:
                loop.run_until_complete(pref.stop())
            except Exception:  # noqa: BLE001 - teardown must not hide results
                pass
    finally:
        loop.close()
        sampler.stop()

    for row in real["per_layer"]:
        result = recorder.results.get(row["layer"])
        row["checksum"] = getattr(result, "checksum", None)
    real["peak_rss_mb"] = sampler.peak()
    return real


def _run_baseline(server: RangeServer, names: list[str],
                  bandwidth_mbps: float) -> dict:
    """Monolithic: download the ENTIRE file, then run real-engine layers."""
    t0 = time.perf_counter()
    status, body, _headers = http_get(server.url, timeout=600)
    download_s = time.perf_counter() - t0
    if status != 200:
        raise RuntimeError(f"baseline full GET expected HTTP 200, got {status}")
    if len(body) != server.size:
        raise RuntimeError(
            f"baseline downloaded {len(body)} bytes, expected {server.size}"
        )

    recorder = _RecordingEngine(REAL)
    _reset_engine()
    t_compute0 = time.perf_counter()
    per_layer: list[dict] = []
    for name in names:
        meta = server.info["layers"][name]
        payload = body[int(meta["start"]):int(meta["end"])]
        c_start = time.perf_counter()
        recorder.run_layer(name, payload)
        c_end = time.perf_counter()
        per_layer.append(
            {
                "layer": name,
                "bytes": len(payload),
                "fetch_s": None,          # no per-layer fetch: one full GET
                "compute_s": c_end - c_start,
                "blocked_s": None,
                "fetch_start_s": None,
                "fetch_end_s": None,
                "compute_start_s": c_start - t0,
                "compute_end_s": c_end - t0,
                "checksum": getattr(recorder.results[name], "checksum", None),
            }
        )
    compute_span = time.perf_counter() - t_compute0
    total = time.perf_counter() - t0
    compute_total = sum(row["compute_s"] for row in per_layer)

    nominal_io = server.size / (bandwidth_mbps * 1_000_000.0 / 8.0)
    masking = max(0.0, min(1.0, 1.0 - download_s / nominal_io))
    return {
        "ttft_s": total,                       # first token needs every layer
        "first_layer_s": download_s + per_layer[0]["compute_s"],
        "total_s": total,
        "io_wait_s": download_s,
        "compute_s": compute_total,
        "slice_s": max(0.0, compute_span - compute_total),
        "masking_ratio": masking,
        "bytes_downloaded": len(body),
        "nominal_io_s": nominal_io,
        "peak_rss_mb": float("nan"),           # sampler not running here
        "per_layer": per_layer,
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def _summary_items(result: dict, bandwidth_mbps: float, url: str,
                   names: list[str], device_note: str) -> list[tuple[str, object]]:
    return [
        ("first_layer_ttft_s", result.get("first_layer_s")),
        ("ttft_s", result.get("ttft_s")),
        ("total_s", result.get("total_s")),
        ("io_wait_s", result.get("io_wait_s")),
        ("compute_s", result.get("compute_s")),
        ("slice_s", result.get("slice_s")),
        ("masking_ratio", result.get("masking_ratio")),
        ("bytes_downloaded", result.get("bytes_downloaded")),
        ("nominal_io_s", result.get("nominal_io_s")),
        ("peak_rss_mb", result.get("peak_rss_mb")),
        ("bandwidth_mbps", bandwidth_mbps),
        ("url", url),
        ("model", "TinyLlama-1.1B-Chat-v1.0 (22 layers, hidden 2048, "
                  "head_dim 64, kv_heads 4)"),
        ("layers", ",".join(names)),
        ("runtime", device_note),
    ]


def _write_csv(path: str, real: dict, baseline: dict, names: list[str],
               bandwidth_mbps: float, url: str, notes: list[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["mode", "engine"] + PER_LAYER_FIELDS + ["checksum", "metric", "value"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for mode, result in (("pipeline", real), ("baseline", baseline)):
            for row in result["per_layer"]:
                writer.writerow(
                    [mode, "real"]
                    + [_cell(row.get(k)) for k in PER_LAYER_FIELDS]
                    + [_cell(row.get("checksum")), "", ""]
                )
            writer.writerow([])
            for key, value in _summary_items(result, bandwidth_mbps, url, names,
                                             notes[0]):
                writer.writerow(
                    [mode, "real"] + [""] * len(PER_LAYER_FIELDS)
                    + ["", key, _cell(value)]
                )
            for note in notes[1:]:
                writer.writerow(
                    [mode, "real"] + [""] * len(PER_LAYER_FIELDS)
                    + ["", "note", note]
                )
            writer.writerow([])
    return path


def _print_layer_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'layer':<18} {'bytes':>12} {'fetch_s':>9} {'blocked_s':>10} "
          f"{'compute_s':>10} {'checksum':>10}")
    for row in rows:
        fetch = "-" if row.get("fetch_s") is None else f"{row['fetch_s']:.4f}"
        blocked = "-" if row.get("blocked_s") is None else f"{row['blocked_s']:.4f}"
        checksum = row.get("checksum")
        checksum_s = "-" if checksum is None else f"{checksum:.5f}"
        print(f"  {row['layer']:<18} {row['bytes']:>12,} {fetch:>9} "
              f"{blocked:>10} {row['compute_s']:>10.4f} {checksum_s:>10}")


def main() -> int:
    bandwidth_mbps = 1000.0  # 1 Gbps loopback: I/O-bound vs real-GEMM compute
    names = _forward_order(REAL.layer_keys)
    cuda_note = (
        f"torch {torch.__version__}; engine device={REAL.device}; "
        f"torch.cuda.is_available()={torch.cuda.is_available()} "
        "(CPU runtime; CUDA is used only if device='cuda' and CUDA is available)"
    )
    print(f"Real engine: {REAL.layer_count} layers | hidden={REAL.hidden_size} | "
          f"head_dim={REAL.head_dim} | kv_heads={REAL.num_kv_heads}")
    print(f"Runtime: {cuda_note}")
    print(f"Model: {MODEL}")

    server = RangeServer(MODEL, host="127.0.0.1", port=0,
                         bandwidth_mbps=bandwidth_mbps, log_requests=False).start()
    try:
        url = server.url
        print(f"RangeServer: {url} (shared throttle {bandwidth_mbps:g} Mbit/s)")

        layers = {
            n: {
                "start": server.info["layers"][n]["start"],
                "end": server.info["layers"][n]["end"],
                "dtype": "BF16",
            }
            for n in names
        }

        # ---------- progressive (streaming) run ----------
        real = _run_progressive(url, layers, names, bandwidth_mbps)
        _print_layer_table("Progressive run [real engine, streaming]", real["per_layer"])
        print(f"\n  rows={len(real['per_layer'])}  "
              f"first-layer TTFT {real['first_layer_s']:.4f}s  "
              f"masking {real['masking_ratio']:.3f}  "
              f"io_wait {real['io_wait_s']:.4f}s  "
              f"compute {real['compute_s']:.4f}s  "
              f"total {real['total_s']:.4f}s  "
              f"peak RSS {real['peak_rss_mb']:.1f} MB")

        # ---------- baseline (monolithic full download, then compute) ----------
        baseline = _run_baseline(server, names, bandwidth_mbps)
        print(f"\nBaseline [real engine, full pre-download] "
              f"{baseline['first_layer_s']:.4f}s first layer "
              f"(download {baseline['io_wait_s']:.4f}s + "
              f"compute {baseline['per_layer'][0]['compute_s']:.4f}s), "
              f"total {baseline['total_s']:.4f}s")

        for result in (real, baseline):
            if len(result["per_layer"]) != REAL.layer_count:
                raise RuntimeError(
                    f"expected {REAL.layer_count} layers, got "
                    f"{len(result['per_layer'])}"
                )
            for row in result["per_layer"]:
                if row.get("checksum") is not None and not math.isfinite(row["checksum"]):
                    raise RuntimeError(f"non-finite checksum at {row['layer']}")

        notes = [
            cuda_note,
            "layer order: numeric 0..21 (checkpoint stores blocks lexicographically)",
            "seq=1 single-token forward pass; per-layer compute from real BF16 weights",
            "masking_ratio = 1 - io_wait / nominal_io, nominal_io = bytes/(bandwidth/8)",
            "local RangeServer + loopback stands in for MinIO/S3 + tc/netem",
        ]
        eff_mbps = baseline["bytes_downloaded"] * 8.0 / baseline["io_wait_s"] / 1e6
        if eff_mbps < bandwidth_mbps:
            notes.append(
                f"measured loopback throughput ~{eff_mbps:.1f} Mbit/s vs "
                f"{bandwidth_mbps:g} Mbit/s nominal (per-chunk pacing overhead); "
                "io_wait >= nominal_io, so masking_ratio clamps to 0"
            )
        _write_csv(OUT_CSV, real, baseline, names, bandwidth_mbps, url, notes)
        size = os.path.getsize(OUT_CSV)
        print(f"\nwrote {OUT_CSV} ({size} bytes)")

        print("\n=== REAL numbers (measured, CPU runtime) ===")
        print(f"  per-layer compute_s : {min(r['compute_s'] for r in real['per_layer']):.4f} "
              f"- {max(r['compute_s'] for r in real['per_layer']):.4f} s "
              f"(total {real['compute_s']:.4f} s over 22 layers)")
        print(f"  first-layer TTFT    : progressive {real['first_layer_s']:.4f} s | "
              f"baseline {baseline['first_layer_s']:.4f} s")
        print(f"  masking_ratio       : progressive {real['masking_ratio']:.4f} | "
              f"baseline {baseline['masking_ratio']:.4f}")
        print(f"  io_wait             : progressive {real['io_wait_s']:.4f} s | "
              f"baseline {baseline['io_wait_s']:.4f} s")
        print(f"  total               : progressive {real['total_s']:.4f} s | "
              f"baseline {baseline['total_s']:.4f} s")
        print(f"  runtime             : CPU only — torch {torch.__version__}, "
              f"torch.cuda.is_available()={torch.cuda.is_available()} "
              "(CUDA is used only when device='cuda' and CUDA is available)")
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
