"""Comprehensive Empirical Benchmark & Ablation Suite for Progressive Layer-Streaming.

Runs genuine PyTorch Llama-architecture decoder forward passes (RoPE + Causal GQA +
SwiGLU MLP + RMSNorm) over HTTP Range requests against a bandwidth-throttled server.

Evaluates 4 core research dimensions:
  1. Bandwidth Sweep: 100, 250, 500, 1000 Mbps (Monolithic vs Progressive Evicting)
  2. Lookahead Depth Ablation: K in {1, 2, 3, 4} (Latency vs Peak Working-Set RSS)
  3. Prompt Length Scaling: seq_len in {16, 64, 128, 256} (I/O-bound -> Compute-balanced)
  4. Detailed Pipeline Gantt & RSS Memory Trajectory (Monolithic vs Progressive Evicting)

Every reported metric is directly measured on this machine — zero synthetic numbers.
"""

from __future__ import annotations

import asyncio
import csv
import gc
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.RealModelComputeEngine import RealModelComputeEngine
from src.storage.RangeServer import RangeServer

MODEL_PATH = ROOT / "models" / "llama_12layer_140m.safetensors"
RESULTS_DIR = ROOT / "results"


class RSSMonitor:
    """High-frequency process RSS sampler (MB) with relative timestamps."""

    def __init__(self, interval: float = 0.04) -> None:
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._t0 = time.perf_counter()
        self.samples.append((0.0, self.proc.memory_info().rss / (1024 * 1024)))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter() - self._t0
            mb = self.proc.memory_info().rss / (1024 * 1024)
            self.samples.append((t, mb))
            self._stop.wait(self.interval)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        t = time.perf_counter() - self._t0
        mb = self.proc.memory_info().rss / (1024 * 1024)
        self.samples.append((t, mb))
        base = self.samples[0][1]
        peak = max(s[1] for s in self.samples)
        return {
            "baseline_rss_mb": round(base, 2),
            "peak_rss_mb": round(peak, 2),
            "delta_rss_mb": round(max(0.0, peak - base), 2),
        }


class EvictingAsyncPrefetcher:
    """Bounded-buffer HTTP Range prefetcher with strict lookahead K and immediate eviction."""

    def __init__(self, url: str, layers: dict[str, dict], lookahead: int = 2) -> None:
        self.url = url
        self.layers = layers
        self.order = list(layers.keys())
        self.lookahead = max(1, int(lookahead))
        self._parsed = urllib.parse.urlparse(url)
        self._queue: dict[str, asyncio.Future] = {}
        self._sem = asyncio.Semaphore(self.lookahead)
        self._tasks: list[asyncio.Task] = []
        self.fetch_meta: dict[str, dict[str, float]] = {}
        self.total_bytes = 0
        self._t0 = 0.0

    def _fetch_sync(self, name: str, start: int, end: int) -> bytes:
        t_start = time.perf_counter()
        conn = http.client.HTTPConnection(self._parsed.hostname, self._parsed.port, timeout=120)
        try:
            conn.request("GET", self._parsed.path, headers={"Range": f"bytes={start}-{end - 1}"})
            resp = conn.getresponse()
            data = resp.read()
        finally:
            conn.close()
        t_end = time.perf_counter()
        self.fetch_meta[name] = {
            "fetch_start_s": t_start - self._t0,
            "fetch_end_s": t_end - self._t0,
            "download_s": t_end - t_start,
            "bytes": len(data),
        }
        self.total_bytes += len(data)
        return data

    async def start(self, t0: float) -> None:
        self._t0 = t0
        loop = asyncio.get_running_loop()
        for name in self.order:
            self._queue[name] = loop.create_future()

        async def _producer() -> None:
            for name in self.order:
                await self._sem.acquire()
                meta = self.layers[name]
                data = await asyncio.to_thread(self._fetch_sync, name, meta["start"], meta["end"])
                if not self._queue[name].done():
                    self._queue[name].set_result(data)

        self._tasks.append(asyncio.create_task(_producer()))

    async def pop(self, name: str) -> bytes:
        """Retrieve layer payload and immediately remove internal reference (evict)."""
        data = await self._queue[name]
        # Evict internal reference so only the active compute layer holds it
        self._queue.pop(name, None)
        return data

    def release_slot(self) -> None:
        """Signal that compute finished and the buffer slot can be reused."""
        self._sem.release()

    async def stop(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._queue.clear()


def _forward_order(keys: list[str]) -> list[str]:
    def idx(name: str) -> int:
        tail = name.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else 1 << 30
    return sorted(keys, key=lambda n: (idx(n), n))


def run_monolithic(
    url: str,
    engine: RealModelComputeEngine,
    layer_names: list[str],
    seq_len: int,
    bandwidth_mbps: float,
) -> dict:
    """Traditional all-or-nothing serverless loading: download entire checkpoint, then compute."""
    gc.collect()
    parsed = urllib.parse.urlparse(url)
    mon = RSSMonitor(interval=0.04)
    mon.start()
    t0 = time.perf_counter()

    # 1. Full blocking checkpoint download
    dl_start = time.perf_counter()
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
    try:
        conn.request("GET", parsed.path)
        resp = conn.getresponse()
        full_blob = resp.read()
    finally:
        conn.close()
    dl_end = time.perf_counter()
    download_s = dl_end - dl_start

    # 2. Sequential forward pass across all layers
    engine._init_hidden(seq_len=seq_len)
    per_layer = []
    compute_total = 0.0
    first_layer_s = 0.0

    for i, name in enumerate(layer_names):
        desc = engine.all_layers_info[name]
        c_start = time.perf_counter()
        payload = full_blob[desc["start"]:desc["end"]]
        res = engine.run_layer(name, payload)
        c_end = time.perf_counter()
        compute_total += (c_end - c_start)
        if i == 0:
            first_layer_s = c_end - t0
        per_layer.append({
            "index": i,
            "layer": name,
            "bytes": desc["end"] - desc["start"],
            "fetch_start_s": round(dl_start - t0, 4),
            "fetch_end_s": round(dl_end - t0, 4),
            "compute_start_s": round(c_start - t0, 4),
            "compute_end_s": round(c_end - t0, 4),
            "compute_s": round(c_end - c_start, 4),
            "checksum": round(float(res.checksum), 5),
        })

    total_s = time.perf_counter() - t0
    rss = mon.stop()
    rss_trace = [(round(t, 3), round(m, 2)) for t, m in mon.samples]
    del full_blob
    gc.collect()

    return {
        "mode": "monolithic",
        "bandwidth_mbps": bandwidth_mbps,
        "seq_len": seq_len,
        "lookahead": 0,
        "first_layer_s": round(first_layer_s, 4),
        "ttft_full_s": round(total_s, 4),
        "download_s": round(download_s, 4),
        "compute_s": round(compute_total, 4),
        "io_wait_s": round(download_s, 4),
        "masking_ratio": 0.0,
        "bytes_downloaded": os.path.getsize(MODEL_PATH),
        **rss,
        "per_layer": per_layer,
        "rss_trace": rss_trace,
    }


def run_progressive(
    url: str,
    engine: RealModelComputeEngine,
    layer_names: list[str],
    seq_len: int,
    bandwidth_mbps: float,
    lookahead: int = 2,
    evict: bool = True,
) -> dict:
    """Asynchronous progressive layer-streaming with bounded lookahead K and layer eviction."""
    gc.collect()
    layers_meta = {
        n: {
            "start": engine.all_layers_info[n]["start"],
            "end": engine.all_layers_info[n]["end"],
        }
        for n in layer_names
    }
    engine._init_hidden(seq_len=seq_len)
    retained = []  # only used if evict=False

    mon = RSSMonitor(interval=0.04)
    mon.start()
    t0 = time.perf_counter()

    async def _exec() -> tuple[list[dict], float, float, float, int]:
        pref = EvictingAsyncPrefetcher(url, layers_meta, lookahead=lookahead)
        await pref.start(t0)
        per_layer = []
        io_wait = 0.0
        compute_total = 0.0
        first_layer_s = 0.0
        try:
            for i, name in enumerate(layer_names):
                w0 = time.perf_counter()
                payload = await pref.pop(name)
                w1 = time.perf_counter()
                blocked = max(0.0, w1 - w0)
                io_wait += blocked

                c0 = time.perf_counter()
                res = await asyncio.to_thread(engine.run_layer, name, payload)
                c1 = time.perf_counter()
                compute_total += (c1 - c0)
                if i == 0:
                    first_layer_s = c1 - t0

                if evict:
                    del payload
                else:
                    retained.append(payload)
                pref.release_slot()

                fm = pref.fetch_meta.get(name, {})
                per_layer.append({
                    "index": i,
                    "layer": name,
                    "bytes": int(fm.get("bytes", 0)),
                    "fetch_start_s": round(float(fm.get("fetch_start_s", 0.0)), 4),
                    "fetch_end_s": round(float(fm.get("fetch_end_s", 0.0)), 4),
                    "download_s": round(float(fm.get("download_s", 0.0)), 4),
                    "blocked_s": round(blocked, 4),
                    "compute_start_s": round(c0 - t0, 4),
                    "compute_end_s": round(c1 - t0, 4),
                    "compute_s": round(c1 - c0, 4),
                    "checksum": round(float(res.checksum), 5),
                })
            return per_layer, first_layer_s, compute_total, io_wait, pref.total_bytes
        finally:
            await pref.stop()

    per_layer, first_layer_s, compute_total, io_wait, total_bytes = asyncio.run(_exec())
    total_s = time.perf_counter() - t0
    rss = mon.stop()
    rss_trace = [(round(t, 3), round(m, 2)) for t, m in mon.samples]
    retained.clear()
    gc.collect()

    # Actual measured network transfer sum across all layers
    total_net_s = sum(r["download_s"] for r in per_layer)
    # Masking ratio = fraction of network transfer hidden behind concurrent compute
    # Exact definition: 1 - (io_wait / total_net_s), clamped to [0, 1]
    masking_ratio = max(0.0, min(1.0, 1.0 - (io_wait / total_net_s))) if total_net_s > 0 else 0.0

    return {
        "mode": "progressive_evict" if evict else "progressive_retain",
        "bandwidth_mbps": bandwidth_mbps,
        "seq_len": seq_len,
        "lookahead": lookahead,
        "first_layer_s": round(first_layer_s, 4),
        "ttft_full_s": round(total_s, 4),
        "download_s": round(total_net_s, 4),
        "compute_s": round(compute_total, 4),
        "io_wait_s": round(io_wait, 4),
        "masking_ratio": round(masking_ratio, 4),
        "bytes_downloaded": total_bytes,
        **rss,
        "per_layer": per_layer,
        "rss_trace": rss_trace,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Run make_transformer_model.py first.")

    engine = RealModelComputeEngine(str(MODEL_PATH))
    layer_names = _forward_order(engine.layer_keys)
    print(f"Loaded {MODEL_PATH.name}: {len(layer_names)} layers, device={engine.device}")

    # 1. Bandwidth sweep (100, 250, 500, 1000 Mbps) at seq_len=128, K=2
    bw_results = []
    gantt_pair = {}
    for bw in (100.0, 250.0, 500.0, 1000.0):
        print(f"\n[Exp 1] Bandwidth = {bw} Mbps (seq_len=128, K=2)...")
        srv = RangeServer(str(MODEL_PATH), host="127.0.0.1", port=0, bandwidth_mbps=bw)
        srv.start()
        try:
            mono = run_monolithic(srv.url, engine, layer_names, seq_len=128, bandwidth_mbps=bw)
            prog = run_progressive(srv.url, engine, layer_names, seq_len=128, bandwidth_mbps=bw, lookahead=2, evict=True)
        finally:
            srv.stop()

        speedup = round(mono["ttft_full_s"] / prog["ttft_full_s"], 3)
        ttft_red_pct = round((1.0 - prog["ttft_full_s"] / mono["ttft_full_s"]) * 100.0, 2)
        first_layer_speedup = round(mono["first_layer_s"] / prog["first_layer_s"], 2)
        rss_save_pct = round((1.0 - prog["delta_rss_mb"] / max(mono["delta_rss_mb"], 1.0)) * 100.0, 2)

        row = {
            "bandwidth_mbps": bw,
            "mono_ttft_s": mono["ttft_full_s"],
            "prog_ttft_s": prog["ttft_full_s"],
            "ttft_reduction_pct": ttft_red_pct,
            "speedup_x": speedup,
            "mono_first_layer_s": mono["first_layer_s"],
            "prog_first_layer_s": prog["first_layer_s"],
            "first_layer_speedup_x": first_layer_speedup,
            "prog_masking_ratio": prog["masking_ratio"],
            "mono_compute_s": mono["compute_s"],
            "prog_compute_s": prog["compute_s"],
            "mono_download_s": mono["download_s"],
            "prog_io_wait_s": prog["io_wait_s"],
            "mono_peak_rss_mb": mono["peak_rss_mb"],
            "prog_peak_rss_mb": prog["peak_rss_mb"],
            "mono_delta_rss_mb": mono["delta_rss_mb"],
            "prog_delta_rss_mb": prog["delta_rss_mb"],
            "rss_delta_reduction_pct": rss_save_pct,
        }
        bw_results.append(row)
        print(
            f"  Mono TTFT={mono['ttft_full_s']:.2f}s | Prog TTFT={prog['ttft_full_s']:.2f}s "
            f"({ttft_red_pct:+.1f}%, {speedup:.2f}x) | Masking={prog['masking_ratio']*100:.1f}% | "
            f"DeltaRSS: {mono['delta_rss_mb']:.1f} vs {prog['delta_rss_mb']:.1f} MB"
        )
        if bw == 250.0:
            gantt_pair = {"monolithic": mono, "progressive": prog}

    # 2. Lookahead depth ablation (K = 1, 2, 3, 4) at 250 Mbps, seq_len=128
    lookahead_results = []
    print("\n[Exp 2] Lookahead Depth Ablation (250 Mbps, seq_len=128)...")
    srv = RangeServer(str(MODEL_PATH), host="127.0.0.1", port=0, bandwidth_mbps=250.0)
    srv.start()
    try:
        for k in (1, 2, 3, 4):
            prog = run_progressive(srv.url, engine, layer_names, seq_len=128, bandwidth_mbps=250.0, lookahead=k, evict=True)
            lookahead_results.append({
                "lookahead_k": k,
                "ttft_full_s": prog["ttft_full_s"],
                "first_layer_s": prog["first_layer_s"],
                "masking_ratio": prog["masking_ratio"],
                "io_wait_s": prog["io_wait_s"],
                "peak_rss_mb": prog["peak_rss_mb"],
                "delta_rss_mb": prog["delta_rss_mb"],
            })
            print(
                f"  K={k}: TTFT={prog['ttft_full_s']:.2f}s | Masking={prog['masking_ratio']*100:.1f}% | "
                f"DeltaRSS={prog['delta_rss_mb']:.1f} MB"
            )
    finally:
        srv.stop()

    # 3. Prompt sequence length scaling (seq_len = 16, 64, 128, 256) at 500 Mbps, K=2
    seq_results = []
    print("\n[Exp 3] Sequence Length Scaling (500 Mbps, K=2)...")
    srv = RangeServer(str(MODEL_PATH), host="127.0.0.1", port=0, bandwidth_mbps=500.0)
    srv.start()
    try:
        for seq in (16, 64, 128, 256):
            mono = run_monolithic(srv.url, engine, layer_names, seq_len=seq, bandwidth_mbps=500.0)
            prog = run_progressive(srv.url, engine, layer_names, seq_len=seq, bandwidth_mbps=500.0, lookahead=2, evict=True)
            seq_results.append({
                "seq_len": seq,
                "mono_ttft_s": mono["ttft_full_s"],
                "prog_ttft_s": prog["ttft_full_s"],
                "speedup_x": round(mono["ttft_full_s"] / prog["ttft_full_s"], 3),
                "ttft_reduction_pct": round((1.0 - prog["ttft_full_s"] / mono["ttft_full_s"]) * 100.0, 2),
                "compute_s": prog["compute_s"],
                "download_s": prog["download_s"],
                "masking_ratio": prog["masking_ratio"],
            })
            print(
                f"  seq={seq:3d}: Mono={mono['ttft_full_s']:.2f}s | Prog={prog['ttft_full_s']:.2f}s | "
                f"Compute={prog['compute_s']:.2f}s | Masking={prog['masking_ratio']*100:.1f}%"
            )
    finally:
        srv.stop()

    # Save CSVs & JSON summary
    with open(RESULTS_DIR / "ablation_bandwidth.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bw_results[0].keys()))
        w.writeheader()
        w.writerows(bw_results)

    with open(RESULTS_DIR / "ablation_lookahead.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lookahead_results[0].keys()))
        w.writeheader()
        w.writerows(lookahead_results)

    with open(RESULTS_DIR / "ablation_seqlen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq_results[0].keys()))
        w.writeheader()
        w.writerows(seq_results)

    summary = {
        "model": MODEL_PATH.name,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2),
        "num_layers": len(layer_names),
        "device": str(engine.device),
        "bandwidth_sweep": bw_results,
        "lookahead_ablation": lookahead_results,
        "seqlen_scaling": seq_results,
        "gantt_250mbps": gantt_pair,
    }
    (RESULTS_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved all results to {RESULTS_DIR / 'ablation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
