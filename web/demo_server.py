"""Live demo server for the real-model streaming engine.

Starts a RangeServer on port 8001 serving models/tinyllama.safetensors with a
throttled link (bandwidth_mbps=1000.0), runs the REAL forward pass in a
background thread (RealModelComputeEngine, auto device), and serves the browser
UI on port 8000.

Run:
  cd /home/bhandi/opencode-test
  python3 web/demo_server.py
  # then open http://127.0.0.1:8000/demo.html
"""

from __future__ import annotations

import asyncio
import http.client
import http.server
import json
import os
import sys
import threading
import time
import urllib.parse
from socketserver import ThreadingMixIn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.storage.RangeServer import RangeServer  # noqa: E402
from src.core.RealModelComputeEngine import RealModelComputeEngine  # noqa: E402
from src.bench.BenchmarkHarness import FallbackPrefetcher, RSSSampler  # noqa: E402

MODEL = os.path.join(ROOT, "models", "tinyllama.safetensors")
WEB = os.path.dirname(os.path.abspath(__file__))
LIVE_PATH = os.path.join(WEB, "live_results.json")
RESULTS_PATH = os.path.join(WEB, "results.json")

BANDWIDTH_MBPS = 1000.0
import torch  # noqa: E402
RUNTIME_STR = f"torch {torch.__version__}; cuda_available={torch.cuda.is_available()}"

REAL = RealModelComputeEngine(MODEL)

# --- shared state -----------------------------------------------------------
LIVE_ROWS: list[dict] = []
FINAL: dict | None = None
STATUS: str = "idle"
RUNNING: bool = False
SCHEDULED: bool = False
RUN_LOCK = threading.Lock()

RANGE_SERVER: RangeServer | None = None


def _forward_order(keys: list[str]) -> list[str]:
    def index(name: str) -> int:
        tail = name.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else 1 << 30

    return sorted(keys, key=lambda n: (index(n), n))


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _reset_state() -> None:
    global LIVE_ROWS, FINAL, STATUS
    LIVE_ROWS = []
    FINAL = None
    STATUS = "running"
    try:
        _write_json(LIVE_PATH, [])
    except Exception:
        pass
    try:
        if os.path.exists(RESULTS_PATH):
            os.remove(RESULTS_PATH)
    except Exception:
        pass


def run_demo() -> None:
    global LIVE_ROWS, FINAL, STATUS, RUNNING

    _reset_state()
    with RUN_LOCK:
        RUNNING = True

    try:
        url = RANGE_SERVER.url if RANGE_SERVER is not None else "http://127.0.0.1:8001/model.safetensors"
        names = _forward_order(REAL.layer_keys)
        layers = {
            n: {
                "start": RANGE_SERVER.info["layers"][n]["start"],
                "end": RANGE_SERVER.info["layers"][n]["end"],
                "dtype": "BF16",
            }
            for n in names
        }

        REAL._init_hidden()

        async def _run() -> None:
            global LIVE_ROWS, FINAL, STATUS
            pref = FallbackPrefetcher(url, layers, lookahead=2, timeout=600)
            sampler = RSSSampler(interval=0.05)
            sampler.start()
            t0 = time.perf_counter()
            io_wait = 0.0
            compute_total = 0.0
            first_layer = None
            try:
                await pref.start()
                for i, name in enumerate(names):
                    want = time.perf_counter()
                    payload = await pref.get(name)
                    ready = time.perf_counter()
                    blocked = max(0.0, ready - want)
                    io_wait += blocked

                    c_start = time.perf_counter()
                    result = await asyncio.to_thread(REAL.run_layer, name, payload)
                    c_end = time.perf_counter()
                    compute_total += c_end - c_start
                    if i == 0:
                        first_layer = c_end - t0

                    ft = pref.fetch_times.get(name)
                    download_s = ft[2] if ft else blocked
                    entry = {
                        "layer": name,
                        "download_s": float(download_s),
                        "compute_s": float(c_end - c_start),
                        "checksum": float(result.checksum),
                        "timestamp": time.time(),
                    }
                    LIVE_ROWS.append(entry)
                    _write_json(LIVE_PATH, list(LIVE_ROWS))

                total = time.perf_counter() - t0
                nbytes = pref.total_bytes
                nominal_io = (nbytes / (BANDWIDTH_MBPS * 1_000_000.0 / 8.0)) if BANDWIDTH_MBPS > 0 else 0.0
                masking = 0.0
                if nominal_io > 0:
                    masking = max(0.0, min(1.0, 1.0 - io_wait / nominal_io))
                sampler.stop()
                peak = sampler.peak()
                FINAL = {
                    "first_layer_s": float(first_layer) if first_layer is not None else None,
                    "total_s": float(total),
                    "compute_s": float(compute_total),
                    "io_wait_s": float(io_wait),
                    "masking_ratio": float(masking),
                    "peak_rss_mb": float(peak),
                    "bytes_downloaded": int(nbytes),
                    "layers": len(names),
                    "runtime": RUNTIME_STR,
                }
                _write_json(RESULTS_PATH, FINAL)
                STATUS = "done"
            finally:
                try:
                    await pref.stop()
                except Exception:
                    pass
                try:
                    sampler.stop()
                except Exception:
                    pass

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_run())
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        STATUS = "error: " + repr(exc)
        import traceback
        traceback.print_exc()
    finally:
        with RUN_LOCK:
            RUNNING = False


def trigger_run() -> bool:
    global SCHEDULED
    with RUN_LOCK:
        if RUNNING or SCHEDULED:
            return False
        SCHEDULED = True

    def worker() -> None:
        global SCHEDULED
        try:
            run_demo()
        finally:
            with RUN_LOCK:
                SCHEDULED = False

    threading.Thread(target=worker, daemon=True, name="DemoRun").start()
    return True


class DemoHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _serve_file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except FileNotFoundError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _proxy(self) -> None:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=600)
            headers = {}
            for k, v in self.headers.items():
                if k.lower() in ("host", "connection", "content-length"):
                    continue
                headers[k] = v
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length > 0 else None
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("connection", "transfer-encoding"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            try:
                self.send_error(502, str(exc))
            except Exception:
                pass

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if path == "/demo.html":
            self._serve_file(os.path.join(WEB, "demo.html"), "text/html; charset=utf-8")
            return
        if path == "/results.json":
            with RUN_LOCK:
                snapshot = {
                    "status": STATUS,
                    "rows": list(LIVE_ROWS),
                    "final": FINAL,
                    "runtime": RUNTIME_STR,
                    "n_layers": REAL.layer_count,
                    "bandwidth_mbps": BANDWIDTH_MBPS,
                }
            self._send_json(snapshot)
            return
        if path == "/live_results.json":
            with RUN_LOCK:
                self._send_json(list(LIVE_ROWS))
            return
        if path == "/run":
            # allow GET trigger too
            started = trigger_run()
            self._send_json({"started": started, "status": STATUS})
            return
        if path == "/status":
            with RUN_LOCK:
                self._send_json({"status": STATUS, "running": RUNNING, "rows": len(LIVE_ROWS)})
            return
        # proxy model route (and anything else) to port 8001
        self._proxy()

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if path == "/run":
            started = trigger_run()
            self._send_json({"started": started, "status": STATUS})
            return
        self._proxy()

    def do_HEAD(self):  # noqa: N802
        self._proxy()


class ThreadingServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    global RANGE_SERVER

    # 1. RangeServer on port 8001
    RANGE_SERVER = RangeServer(
        MODEL, host="127.0.0.1", port=8001, bandwidth_mbps=BANDWIDTH_MBPS, log_requests=False
    ).start()
    print(f"RangeServer: {RANGE_SERVER.url} (throttle {BANDWIDTH_MBPS:g} Mbit/s)", flush=True)

    # 2. HTTP server on port 8000
    httpd = ThreadingServer(("127.0.0.1", 8000), DemoHandler)
    print("Demo UI: http://127.0.0.1:8000/demo.html", flush=True)

    # 3. auto-run demo in background
    trigger_run()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        httpd.server_close()
        if RANGE_SERVER is not None:
            RANGE_SERVER.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())