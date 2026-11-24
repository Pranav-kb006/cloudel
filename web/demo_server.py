"""Enterprise Cloud Inference Dashboard Backend for Progressive Layer-Streaming Engine.

Serves:
  - Light-first multi-view SPA (web/demo.html + web/static/styles.css + web/static/app.js)
  - Real backend APIs:
      GET  /api/status        -> Live experiment state, experiment_id, runtime/device/MinIO status
      GET  /api/models        -> Real discovered safetensors checkpoints with parsed header metadata
      GET  /api/history       -> Real measured history (benchmark runs + persisted live runs)
      GET  /api/history.csv   -> Downloadable CSV export of experiment history
      GET  /api/ablation      -> Full measured ablation_summary.json
      GET  /api/logs          -> Real structured system logs (timestamp, level, component, exp_id, msg)
      POST /api/run           -> Trigger real Monolithic / Progressive / Side-by-Side execution
      POST /api/stop          -> Request graceful cancellation of active run
      GET  /static/<file>     -> CSS / JS assets
      GET  /charts/<file>     -> 300-DPI measured Matplotlib charts

Run:
  python web/demo_server.py
  # Open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import csv
import gc
import http.client
import http.server
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from socketserver import ThreadingMixIn

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bench.ablation_suite import EvictingAsyncPrefetcher, RSSMonitor, _forward_order
from src.core.RealModelComputeEngine import RealModelComputeEngine
from src.storage.RangeServer import RangeServer
from src.storage.s3_adapter import check_minio_available
from src.storage.safetensors_ranges import parse_safetensors

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
HISTORY_FILE = RESULTS_DIR / "experiment_history.json"

STATE_LOCK = threading.Lock()
ENGINE_CACHE: dict[str, RealModelComputeEngine] = {}
LOGS: deque[dict[str, str]] = deque(maxlen=600)
STOP_EVENT = threading.Event()
EXP_COUNTER = 0


def emit_log(level: str, component: str, message: str, experiment_id: str = "SYSTEM") -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": level.upper(),
        "component": component,
        "experiment_id": experiment_id,
        "message": message,
    }
    with STATE_LOCK:
        LOGS.append(entry)


def discover_models() -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if not MODELS_DIR.exists():
        return found
    for p in sorted(MODELS_DIR.glob("*.safetensors")):
        try:
            info = parse_safetensors(str(p))
            layers = [k for k in info.get("layers", {}) if k.startswith("model.layers.")]
            is_synth = "synthetic" in p.name.lower()
            first_tensor = next(iter(info.get("tensors", {}).values()), {})
            dtype = first_tensor.get("dtype", "BF16")
            hidden = 0
            for tname, tmeta in info.get("tensors", {}).items():
                if tname.endswith("q_proj.weight") and len(tmeta.get("shape", [])) == 2:
                    hidden = int(tmeta["shape"][1])
                    break
            found.append({
                "name": p.name,
                "path": str(p),
                "architecture": "Synthetic FP32 Benchmark" if is_synth else "Llama Decoder (RoPE + GQA + SwiGLU)",
                "num_layers": len(layers) if layers else len(info.get("layers", {})),
                "hidden_size": hidden,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                "size_bytes": p.stat().st_size,
                "format": "SafeTensors",
                "dtype": dtype,
                "storage": "Local Object Store / RangeServer",
                "available": True,
                "runnable": not is_synth,
            })
        except Exception as exc:
            emit_log("WARNING", "ModelRegistry", f"Failed to inspect {p.name}: {exc}")
    return found


AVAILABLE_MODELS = discover_models()
RUNNABLE_MODELS = [m for m in AVAILABLE_MODELS if m.get("runnable")]
DEFAULT_MODEL = (
    next((str(m["path"]) for m in RUNNABLE_MODELS if "llama_12layer" in str(m["name"])), None)
    or (str(RUNNABLE_MODELS[0]["path"]) if RUNNABLE_MODELS else str(MODELS_DIR / "llama_12layer_140m.safetensors"))
)


def get_engine(model_path: str) -> RealModelComputeEngine:
    if model_path not in ENGINE_CACHE:
        emit_log("INFO", "ComputeEngine", f"Initializing RealModelComputeEngine for {os.path.basename(model_path)}")
        ENGINE_CACHE[model_path] = RealModelComputeEngine(model_path)
    return ENGINE_CACHE[model_path]


def load_history() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ab_path = RESULTS_DIR / "ablation_summary.json"
    if ab_path.exists():
        try:
            ab = json.loads(ab_path.read_text(encoding="utf-8"))
            model_name = ab.get("model", "llama_12layer_140m.safetensors")
            for idx, r in enumerate(ab.get("bandwidth_sweep", []), start=1):
                records.append({
                    "experiment_id": f"BENCH-BW-{idx:02d}",
                    "source": "benchmark",
                    "model": model_name,
                    "bandwidth_mbps": r["bandwidth_mbps"],
                    "lookahead": 2,
                    "seq_len": 128,
                    "mode": "compare",
                    "mono_ttft_s": r["mono_ttft_s"],
                    "prog_ttft_s": r["prog_ttft_s"],
                    "ttft_reduction_pct": r["ttft_reduction_pct"],
                    "speedup_x": r["speedup_x"],
                    "first_layer_speedup_x": r["first_layer_speedup_x"],
                    "masking_ratio": r["prog_masking_ratio"],
                    "mono_delta_rss_mb": r["mono_delta_rss_mb"],
                    "prog_delta_rss_mb": r["prog_delta_rss_mb"],
                    "rss_delta_reduction_pct": r["rss_delta_reduction_pct"],
                    "status": "completed",
                    "timestamp": "2026-11-24T11:59:00Z",
                })
            for idx, r in enumerate(ab.get("lookahead_ablation", []), start=1):
                records.append({
                    "experiment_id": f"BENCH-LA-{idx:02d}",
                    "source": "benchmark",
                    "model": model_name,
                    "bandwidth_mbps": 250.0,
                    "lookahead": r["lookahead_k"],
                    "seq_len": 128,
                    "mode": "progressive",
                    "mono_ttft_s": None,
                    "prog_ttft_s": r["ttft_full_s"],
                    "ttft_reduction_pct": None,
                    "speedup_x": None,
                    "first_layer_speedup_x": None,
                    "masking_ratio": r["masking_ratio"],
                    "mono_delta_rss_mb": None,
                    "prog_delta_rss_mb": r["delta_rss_mb"],
                    "rss_delta_reduction_pct": None,
                    "status": "completed",
                    "timestamp": "2026-11-24T11:59:30Z",
                })
        except Exception as exc:
            emit_log("WARNING", "History", f"Failed to load ablation history: {exc}")

    if HISTORY_FILE.exists():
        try:
            live_recs = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(live_recs, list):
                records.extend(live_recs)
        except Exception:
            pass
    return records


def append_live_history(rec: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if HISTORY_FILE.exists():
        try:
            existing = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.insert(0, rec)
    HISTORY_FILE.write_text(json.dumps(existing[:100], indent=2), encoding="utf-8")


STATE: dict[str, object] = {
    "status": "idle",
    "phase": "ready",
    "experiment_id": "BENCH-BW-02",
    "started_at": None,
    "elapsed_s": 0.0,
    "error": None,
    "config": {
        "model": os.path.basename(DEFAULT_MODEL),
        "bandwidth_mbps": 250.0,
        "lookahead": 2,
        "seq_len": 128,
        "evict": True,
        "mode": "compare",
    },
    "runtime": {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "minio_available": check_minio_available(),
        "minio_endpoint": os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
        "storage_backend": "MinIO S3 + RangeServer" if check_minio_available() else "HTTP RangeServer (RFC 7233)",
        "models": AVAILABLE_MODELS,
    },
    "monolithic": None,
    "progressive": None,
    "comparison": None,
    "live_layers": [],
    "active_layer": None,
    "prefetching_layer": None,
    "buffer_occupancy": 0,
    "eviction_count": 0,
    "progress": 0.0,
}

RUNNING = False


def _run_monolithic_live(
    url: str, engine: RealModelComputeEngine, layer_names: list[str],
    model_path: str, seq_len: int, bandwidth_mbps: float, exp_id: str,
) -> dict:
    gc.collect()
    parsed = urllib.parse.urlparse(url)
    mon = RSSMonitor(interval=0.05)
    mon.start()
    t0 = time.perf_counter()

    emit_log("INFO", "MonolithicLoader", f"Starting blocking full-checkpoint download ({bandwidth_mbps} Mbps)", exp_id)
    with STATE_LOCK:
        STATE["phase"] = "monolithic_download"
        STATE["live_layers"] = []
        STATE["active_layer"] = "FULL_DOWNLOAD"

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
    emit_log("INFO", "MonolithicLoader", f"Completed full download: {len(full_blob)/1048576:.2f} MiB in {download_s:.2f}s", exp_id)

    with STATE_LOCK:
        STATE["phase"] = "monolithic_compute"

    engine._init_hidden(seq_len=seq_len)
    per_layer = []
    compute_total = 0.0
    first_layer_s = 0.0

    for i, name in enumerate(layer_names):
        if STOP_EVENT.is_set():
            raise RuntimeError("Experiment stopped by user")
        desc = engine.all_layers_info[name]
        c_start = time.perf_counter()
        payload = full_blob[desc["start"]:desc["end"]]
        res = engine.run_layer(name, payload)
        c_end = time.perf_counter()
        compute_total += (c_end - c_start)
        if i == 0:
            first_layer_s = c_end - t0
        row = {
            "index": i,
            "layer": name,
            "bytes": desc["end"] - desc["start"],
            "fetch_start_s": round(dl_start - t0, 4),
            "fetch_end_s": round(dl_end - t0, 4),
            "download_s": round(download_s, 4),
            "blocked_s": round(download_s if i == 0 else 0.0, 4),
            "compute_start_s": round(c_start - t0, 4),
            "compute_end_s": round(c_end - t0, 4),
            "compute_s": round(c_end - c_start, 4),
            "checksum": round(float(res.checksum), 5),
        }
        per_layer.append(row)
        with STATE_LOCK:
            STATE["live_layers"] = list(per_layer)
            STATE["active_layer"] = name
            STATE["elapsed_s"] = round(time.perf_counter() - t0, 2)
            STATE["progress"] = round(0.5 * (i + 1) / len(layer_names), 3)

    total_s = time.perf_counter() - t0
    rss = mon.stop()
    rss_trace = [[round(t, 3), round(max(0.0, m - rss["baseline_rss_mb"]), 2)] for t, m in mon.samples]
    del full_blob
    gc.collect()
    emit_log("INFO", "MonolithicLoader", f"Monolithic pass finished: TTFT={total_s:.2f}s, Peak dRSS={rss['delta_rss_mb']:.1f} MB", exp_id)

    return {
        "mode": "monolithic",
        "bandwidth_mbps": bandwidth_mbps,
        "seq_len": seq_len,
        "first_layer_s": round(first_layer_s, 4),
        "ttft_full_s": round(total_s, 4),
        "download_s": round(download_s, 4),
        "compute_s": round(compute_total, 4),
        "io_wait_s": round(download_s, 4),
        "masking_ratio": 0.0,
        "bytes_downloaded": os.path.getsize(model_path),
        "http_requests": 1,
        **rss,
        "per_layer": per_layer,
        "rss_trace": rss_trace,
    }


def _run_progressive_live(
    url: str, engine: RealModelComputeEngine, layer_names: list[str],
    seq_len: int, bandwidth_mbps: float, lookahead: int, evict: bool,
    exp_id: str, base_progress: float = 0.0, progress_span: float = 1.0,
) -> dict:
    gc.collect()
    layers_meta = {
        n: {
            "start": engine.all_layers_info[n]["start"],
            "end": engine.all_layers_info[n]["end"],
        }
        for n in layer_names
    }
    engine._init_hidden(seq_len=seq_len)
    retained = []

    emit_log(
        "INFO", "AsyncPrefetcher",
        f"Starting progressive stream: {len(layer_names)} layers, K={lookahead}, seq={seq_len}, evict={evict}, bw={bandwidth_mbps} Mbps",
        exp_id,
    )
    with STATE_LOCK:
        STATE["phase"] = "progressive_streaming"
        STATE["live_layers"] = []
        STATE["eviction_count"] = 0

    mon = RSSMonitor(interval=0.05)
    mon.start()
    t0 = time.perf_counter()

    async def _exec():
        pref = EvictingAsyncPrefetcher(url, layers_meta, lookahead=lookahead)
        await pref.start(t0)
        per_layer = []
        io_wait = 0.0
        compute_total = 0.0
        first_layer_s = 0.0
        evictions = 0
        try:
            for i, name in enumerate(layer_names):
                if STOP_EVENT.is_set():
                    raise RuntimeError("Experiment stopped by user")
                next_name = layer_names[i + 1] if i + 1 < len(layer_names) else None
                with STATE_LOCK:
                    STATE["active_layer"] = name
                    STATE["prefetching_layer"] = next_name
                    STATE["buffer_occupancy"] = min(lookahead, len(layer_names) - i)

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
                    evictions += 1
                else:
                    retained.append(payload)
                pref.release_slot()

                fm = pref.fetch_meta.get(name, {})
                row = {
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
                }
                per_layer.append(row)
                emit_log(
                    "DEBUG", "PipelineDriver",
                    f"{name}: fetch={row['download_s']:.3f}s wait={blocked:.3f}s compute={row['compute_s']:.3f}s |h|={row['checksum']:.4f}"
                    + (" [evicted]" if evict else ""),
                    exp_id,
                )
                with STATE_LOCK:
                    STATE["live_layers"] = list(per_layer)
                    STATE["eviction_count"] = evictions
                    STATE["elapsed_s"] = round(time.perf_counter() - t0, 2)
                    STATE["progress"] = round(base_progress + progress_span * (i + 1) / len(layer_names), 3)
            return per_layer, first_layer_s, compute_total, io_wait, pref.total_bytes, evictions
        finally:
            await pref.stop()

    per_layer, first_layer_s, compute_total, io_wait, total_bytes, evictions = asyncio.run(_exec())
    total_s = time.perf_counter() - t0
    rss = mon.stop()
    rss_trace = [[round(t, 3), round(max(0.0, m - rss["baseline_rss_mb"]), 2)] for t, m in mon.samples]
    retained.clear()
    gc.collect()

    total_net_s = sum(r["download_s"] for r in per_layer)
    masking_ratio = max(0.0, min(1.0, 1.0 - (io_wait / total_net_s))) if total_net_s > 0 else 0.0
    emit_log(
        "INFO", "PipelineDriver",
        f"Progressive pass finished: TTFT={total_s:.2f}s, Masking={masking_ratio*100:.1f}%, Peak dRSS={rss['delta_rss_mb']:.1f} MB, Evictions={evictions}",
        exp_id,
    )

    return {
        "mode": "progressive_evict" if evict else "progressive_retain",
        "bandwidth_mbps": bandwidth_mbps,
        "seq_len": seq_len,
        "lookahead": lookahead,
        "evict": evict,
        "eviction_count": evictions,
        "first_layer_s": round(first_layer_s, 4),
        "ttft_full_s": round(total_s, 4),
        "download_s": round(total_net_s, 4),
        "compute_s": round(compute_total, 4),
        "io_wait_s": round(io_wait, 4),
        "masking_ratio": round(masking_ratio, 4),
        "bytes_downloaded": total_bytes,
        "http_requests": len(per_layer),
        **rss,
        "per_layer": per_layer,
        "rss_trace": rss_trace,
    }


def _worker_run(cfg: dict, exp_id: str) -> None:
    global RUNNING
    t_run_start = time.perf_counter()
    try:
        model_name = str(cfg.get("model") or os.path.basename(DEFAULT_MODEL))
        model_path = str(MODELS_DIR / model_name)
        if not os.path.exists(model_path):
            model_path = DEFAULT_MODEL
        bw = max(25.0, min(2000.0, float(cfg.get("bandwidth_mbps", 250.0))))
        lookahead = max(1, min(6, int(cfg.get("lookahead", 2))))
        seq_len = max(1, min(512, int(cfg.get("seq_len", 128))))
        evict = bool(cfg.get("evict", True))
        mode = str(cfg.get("mode", "compare"))

        engine = get_engine(model_path)
        layer_names = _forward_order(engine.layer_keys)

        srv = RangeServer(model_path, host="127.0.0.1", port=0, bandwidth_mbps=bw)
        srv.start()
        emit_log("INFO", "RangeServer", f"Bound throttled HTTP RangeServer at {srv.url} ({bw} Mbps)", exp_id)
        try:
            mono = None
            prog = None
            if mode in ("compare", "monolithic"):
                mono = _run_monolithic_live(srv.url, engine, layer_names, model_path, seq_len, bw, exp_id)
                with STATE_LOCK:
                    STATE["monolithic"] = mono
            if mode in ("compare", "progressive"):
                base_p = 0.5 if mode == "compare" else 0.0
                span_p = 0.5 if mode == "compare" else 1.0
                prog = _run_progressive_live(srv.url, engine, layer_names, seq_len, bw, lookahead, evict, exp_id, base_p, span_p)
                with STATE_LOCK:
                    STATE["progressive"] = prog

            comp = None
            if mono and prog:
                comp = {
                    "ttft_reduction_pct": round((1.0 - prog["ttft_full_s"] / mono["ttft_full_s"]) * 100.0, 2),
                    "speedup_x": round(mono["ttft_full_s"] / prog["ttft_full_s"], 3),
                    "first_layer_speedup_x": round(mono["first_layer_s"] / prog["first_layer_s"], 2),
                    "masking_pct": round(prog["masking_ratio"] * 100.0, 2),
                    "rss_delta_saved_mb": round(max(0.0, mono["delta_rss_mb"] - prog["delta_rss_mb"]), 2),
                    "rss_delta_reduction_pct": round(
                        (1.0 - prog["delta_rss_mb"] / max(mono["delta_rss_mb"], 1.0)) * 100.0, 2
                    ),
                }

            append_live_history({
                "experiment_id": exp_id,
                "source": "live",
                "model": os.path.basename(model_path),
                "bandwidth_mbps": bw,
                "lookahead": lookahead,
                "seq_len": seq_len,
                "mode": mode,
                "mono_ttft_s": mono["ttft_full_s"] if mono else None,
                "prog_ttft_s": prog["ttft_full_s"] if prog else None,
                "ttft_reduction_pct": comp["ttft_reduction_pct"] if comp else None,
                "speedup_x": comp["speedup_x"] if comp else None,
                "first_layer_speedup_x": comp["first_layer_speedup_x"] if comp else None,
                "masking_ratio": prog["masking_ratio"] if prog else 0.0,
                "mono_delta_rss_mb": mono["delta_rss_mb"] if mono else None,
                "prog_delta_rss_mb": prog["delta_rss_mb"] if prog else None,
                "rss_delta_reduction_pct": comp["rss_delta_reduction_pct"] if comp else None,
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

            with STATE_LOCK:
                STATE["comparison"] = comp
                STATE["status"] = "done"
                STATE["phase"] = "completed"
                STATE["active_layer"] = None
                STATE["prefetching_layer"] = None
                STATE["buffer_occupancy"] = 0
                STATE["elapsed_s"] = round(time.perf_counter() - t_run_start, 2)
                STATE["progress"] = 1.0
        finally:
            srv.stop()
            emit_log("INFO", "RangeServer", "Stopped ephemeral RangeServer", exp_id)
    except Exception as exc:
        emit_log("ERROR", "ExperimentRunner", str(exc), exp_id)
        with STATE_LOCK:
            STATE["status"] = "error"
            STATE["phase"] = "failed"
            STATE["error"] = str(exc)
            STATE["active_layer"] = None
            STATE["prefetching_layer"] = None
    finally:
        STOP_EVENT.clear()
        with STATE_LOCK:
            RUNNING = False


def trigger_run(cfg: dict) -> tuple[bool, str]:
    global RUNNING, EXP_COUNTER
    with STATE_LOCK:
        if RUNNING:
            return False, str(STATE["experiment_id"])
        RUNNING = True
        STOP_EVENT.clear()
        EXP_COUNTER += 1
        exp_id = f"EXP-{datetime.now().strftime('%H%M%S')}-{EXP_COUNTER:02d}"
        STATE["status"] = "running"
        STATE["phase"] = "initializing"
        STATE["experiment_id"] = exp_id
        STATE["started_at"] = datetime.now(timezone.utc).isoformat()
        STATE["elapsed_s"] = 0.0
        STATE["error"] = None
        STATE["progress"] = 0.02
        STATE["config"] = dict(cfg)
        if cfg.get("mode") == "compare":
            STATE["monolithic"] = None
            STATE["progressive"] = None
            STATE["comparison"] = None
    threading.Thread(target=_worker_run, args=(dict(cfg), exp_id), daemon=True).start()
    return True, exp_id


class DemoHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _send_bytes(self, raw: bytes, content_type: str, status: int = 200, extra_headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, obj: object, status: int = 200) -> None:
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", status)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        self._send_bytes(path.read_bytes(), content_type)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/demo.html", "/index.html"):
            self._send_file(WEB_DIR / "demo.html", "text/html; charset=utf-8")
            return
        if path == "/api/status":
            with STATE_LOCK:
                snap = dict(STATE)
            self._send_json(snap)
            return
        if path == "/api/models":
            self._send_json({"models": discover_models()})
            return
        if path == "/api/history":
            self._send_json({"history": load_history()})
            return
        if path == "/api/history.csv":
            rows = load_history()
            buf = io.StringIO()
            if rows:
                w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            self._send_bytes(
                buf.getvalue().encode("utf-8"),
                "text/csv; charset=utf-8",
                extra_headers={"Content-Disposition": "attachment; filename=experiment_history.csv"},
            )
            return
        if path == "/api/logs":
            with STATE_LOCK:
                logs_list = list(LOGS)
            self._send_json({"logs": logs_list})
            return
        if path == "/api/ablation":
            ab_path = RESULTS_DIR / "ablation_summary.json"
            if ab_path.exists():
                self._send_file(ab_path, "application/json; charset=utf-8")
            else:
                self._send_json({"available": False}, 200)
            return
        if path.startswith("/static/"):
            fname = os.path.basename(path)
            ctype = "text/css; charset=utf-8" if fname.endswith(".css") else "application/javascript; charset=utf-8"
            self._send_file(STATIC_DIR / fname, ctype)
            return
        if path.startswith("/charts/"):
            fname = os.path.basename(path)
            self._send_file(RESULTS_DIR / "charts" / fname, "image/png")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                cfg = json.loads(body.decode("utf-8"))
            except Exception:
                cfg = {}
            started, exp_id = trigger_run(cfg)
            self._send_json({"started": started, "experiment_id": exp_id}, 200 if started else 409)
            return
        if parsed.path == "/api/stop":
            STOP_EVENT.set()
            emit_log("WARNING", "ExperimentRunner", "Stop requested by operator", str(STATE.get("experiment_id", "SYSTEM")))
            self._send_json({"stopping": True})
            return
        self.send_error(404)


class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    emit_log("INFO", "Server", f"Initialized Progressive AI Cloud Infrastructure Dashboard on port {port}")
    emit_log("INFO", "ModelRegistry", f"Discovered {len(AVAILABLE_MODELS)} safetensors checkpoints ({len(RUNNABLE_MODELS)} runnable)")
    print(f"Progressive AI Cloud Infrastructure Dashboard -> http://127.0.0.1:{port}")
    srv = ThreadedHTTPServer(("0.0.0.0", port), DemoHandler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
