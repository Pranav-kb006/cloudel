"""Local HTTP Range server with a simulated shared-link bandwidth throttle.

STATUS:
  BUILT: stdlib ``ThreadingHTTPServer`` serving a safetensors file with RFC 7233
         single-range support (206 + Content-Range, 416 for invalid ranges,
         HEAD, Accept-Ranges: bytes) and a configurable *shared* bandwidth
         throttle applied by chunked sleep pacing.  Importable ``RangeServer``
         class for in-process use and a ``python3 -m`` CLI.
  TBD:   this is a local stand-in for MinIO/S3 + ``tc``/``netem``.  It has not
         been validated against a real object store or kernel traffic shaping.
  PACKAGES: standard library only.

INTEGRATION:
  The benchmark harness imports ``RangeServer`` and starts it on an ephemeral
  port; ``run.sh`` starts it as a separate process and passes ``--url`` to the
  harness.  The file URL is ``http://<host>:<port>/model.safetensors``
  (``/`` and ``/<basename>`` are also accepted).

THROTTLE SEMANTICS (honesty note):
  The limiter is *global to the server*, so concurrent range requests share the
  configured link rate instead of each getting a private full-rate pipe.  This
  models a single client link shaped by ``tc``; it is deliberately conservative
  for prefetch benchmarks.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable
from urllib.parse import urlsplit

try:  # package import (python3 -m src.storage.RangeServer)
    from .safetensors_ranges import parse_safetensors
except ImportError:  # direct script execution
    from safetensors_ranges import parse_safetensors  # type: ignore


class RangeError(ValueError):
    """Invalid or unsupported Range header."""


class BandwidthLimiter:
    """Global token-pacing limiter shared by all responses.

    ``consume(n)`` blocks until ``n`` bytes are allowed at ``bytes_per_sec``
    across the whole server.  A rate of 0 (or negative) disables throttling.
    """

    def __init__(self, bytes_per_sec: float):
        self.bytes_per_sec = float(bytes_per_sec or 0.0)
        self._lock = threading.Lock()
        self._next_free = 0.0

    def consume(self, nbytes: int) -> None:
        if self.bytes_per_sec <= 0 or nbytes <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._next_free < now:
                self._next_free = now
            self._next_free += nbytes / self.bytes_per_sec
            delay = self._next_free - now
        # Avoid sub-15ms sleep quantization on Windows by letting small credits
        # accumulate until delay >= 0.018s.
        if delay >= 0.018:
            time.sleep(delay)


def parse_range_header(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range.

    Returns ``(start, end_inclusive)`` or ``None`` when no Range was supplied.
    Raises :class:`RangeError` for malformed, multi-range, or unsatisfiable
    requests (the caller answers 416).
    """
    if value is None:
        return None
    value = value.strip()
    if not value.lower().startswith("bytes="):
        raise RangeError("only 'bytes=' ranges are supported")
    spec = value[6:].strip()
    if not spec:
        raise RangeError("empty range spec")
    if "," in spec:
        raise RangeError("multiple ranges are not supported")
    if "-" not in spec:
        raise RangeError("range must contain '-'")
    first, last = spec.split("-", 1)
    first, last = first.strip(), last.strip()

    if first == "":
        # suffix range: bytes=-N -> last N bytes
        if last == "":
            raise RangeError("empty suffix range")
        try:
            suffix = int(last)
        except ValueError as exc:
            raise RangeError("non-integer suffix") from exc
        if suffix <= 0:
            raise RangeError("suffix length must be positive")
        start = max(0, size - suffix)
        end = size - 1
    else:
        try:
            start = int(first)
            end = int(last) if last else size - 1
        except ValueError as exc:
            raise RangeError("non-integer range") from exc
        if start < 0 or end < start:
            raise RangeError("invalid range bounds")
        if start >= size:
            raise RangeError("range start beyond end of file")
        end = min(end, size - 1)

    if size <= 0 or start > end:
        raise RangeError("unsatisfiable range")
    return start, end


class RangeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RangeServer/1.0"

    # -- helpers ---------------------------------------------------------
    @property
    def rs(self) -> "RangeServer":
        return self.server.range_server  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # Silence default per-request logging; we emit structured logs instead.
        if getattr(self.rs, "log_requests", False):
            sys.stderr.write("[RangeServer] " + (fmt % args) + "\n")

    def _route_path(self) -> str:
        return urlsplit(self.path).path or "/"

    def _is_model_route(self, route: str) -> bool:
        base = os.path.basename(self.rs.path)
        return route in ("/", "/model.safetensors", "/" + base)

    def _send_simple(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    # -- HTTP verbs ------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(head_only=False)

    def _serve(self, head_only: bool) -> None:
        rs = self.rs
        route = self._route_path()

        if route == "/health":
            self._send_simple(HTTPStatus.OK, b"ok\n")
            return
        if not self._is_model_route(route):
            self._send_simple(HTTPStatus.NOT_FOUND, b"not found\n")
            return

        started = time.perf_counter()
        try:
            parsed = parse_range_header(self.headers.get("Range"), rs.size)
        except RangeError as exc:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{rs.size}")
            self.send_header("Content-Length", "0")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Connection", "close")
            self.end_headers()
            self._log_request("GET", None, HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                              0, started, note=str(exc))
            return

        if parsed is None:
            start, end = 0, rs.size - 1
            status = HTTPStatus.OK
            is_partial = False
        else:
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT
            is_partial = True

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{rs.size}")
        self.send_header("Connection", "close")
        self.end_headers()

        if head_only:
            self._log_request("HEAD", (start, end), status, 0, started)
            return

        sent = 0
        try:
            with open(rs.path, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    want = min(rs.chunk_size, remaining)
                    chunk = fh.read(want)
                    if not chunk:
                        break
                    # Pace *before* writing so the socket never buffers a large
                    # burst that would hide the configured bandwidth.
                    rs.limiter.consume(len(chunk))
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    sent += len(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            self._log_request("GET", (start, start + sent - 1 if sent else start),
                              status, sent, started, note="client-disconnected")
            return
        except OSError as exc:
            self._log_request("GET", (start, start + sent - 1 if sent else start),
                              status, sent, started, note=f"os-error: {exc}")
            return

        self._log_request("GET", (start, start + sent - 1 if sent else start),
                          status, sent, started)

    def _log_request(
        self,
        method: str,
        byte_range: tuple[int, int] | None,
        status: int,
        nbytes: int,
        started: float,
        note: str = "",
    ) -> None:
        rs = self.rs
        if not getattr(rs, "log_requests", False):
            return
        duration = time.perf_counter() - started
        if byte_range is None:
            rng = "-"
            layers = "-"
        else:
            rng = f"{byte_range[0]}-{byte_range[1]}"
            layers = ",".join(rs.layers_for_range(byte_range[0], byte_range[1])) or "-"
        suffix = f" note={note}" if note else ""
        sys.stderr.write(
            f"[RangeServer] {method} {self.path} range={rng} status={status} "
            f"bytes={nbytes} dur={duration:.4f}s layers={layers}{suffix}\n"
        )
        sys.stderr.flush()


class RangeServer:
    """Importable server wrapper.

    Parameters
    ----------
    path:
        Safetensors file to serve.
    host / port:
        Bind address.  ``port=0`` picks an ephemeral port (useful in tests).
    bandwidth_mbps:
        Simulated link rate in megabits/s.  ``0`` disables throttling.
    log_requests:
        Emit one structured line per request (range, bytes, duration, layers).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        host: str = "127.0.0.1",
        port: int = 8765,
        bandwidth_mbps: float = 8.0,
        log_requests: bool = False,
    ):
        self.path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"RangeServer: file not found: {self.path}")
        self.host = host
        self.port = int(port)
        self.bandwidth_mbps = float(bandwidth_mbps)
        self.bytes_per_sec = (
            self.bandwidth_mbps * 1_000_000.0 / 8.0 if self.bandwidth_mbps > 0 else 0.0
        )
        self.limiter = BandwidthLimiter(self.bytes_per_sec)
        self.log_requests = bool(log_requests)

        self.info = parse_safetensors(self.path)
        self.size = int(self.info["size"])
        self.layer_items = [
            (name, int(meta["start"]), int(meta["end"]))
            for name, meta in self.info["layers"].items()
        ]
        # Chunk sizing: ~25 writes/s at the configured rate, up to 2 MiB so
        # high-bandwidth links (100-1000 Mbps) avoid OS sleep-timer overhead.
        if self.bytes_per_sec > 0:
            self.chunk_size = int(max(4096, min(2 * 1024 * 1024, self.bytes_per_sec / 25.0)))
        else:
            self.chunk_size = 65536

        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "RangeServer":
        self.httpd = ThreadingHTTPServer((self.host, self.port), RangeRequestHandler)
        self.httpd.daemon_threads = True
        self.httpd.range_server = self  # type: ignore[attr-defined]
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, name="RangeServer", daemon=True
        )
        self.thread.start()
        return self

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None

    def __enter__(self) -> "RangeServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- helpers ---------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/model.safetensors"

    def layers_for_range(self, start: int, end: int) -> Iterable[str]:
        """Layer keys overlapping the inclusive byte range ``[start, end]``."""
        if end < start:
            return []
        return [name for name, ls, le in self.layer_items if ls <= end and le > start]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serve a safetensors file over HTTP with Range support "
        "and a simulated shared bandwidth throttle."
    )
    p.add_argument("--file", required=True, help="safetensors file to serve")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="bind port (default 8765, 0=ephemeral)")
    p.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=8.0,
        help="simulated link rate in Mbit/s shared across requests (0 = unlimited)",
    )
    p.add_argument("--log", action="store_true", help="log one line per request")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        server = RangeServer(
            args.file,
            host=args.host,
            port=args.port,
            bandwidth_mbps=args.bandwidth_mbps,
            log_requests=args.log,
        )
        server.start()
    except Exception as exc:  # noqa: BLE001 - CLI should print a clean error
        sys.stderr.write(f"RangeServer failed to start: {type(exc).__name__}: {exc}\n")
        return 1

    rate = "unlimited" if server.bytes_per_sec <= 0 else f"{args.bandwidth_mbps:g} Mbit/s shared"
    print(f"RangeServer serving {server.path}", flush=True)
    print(f"URL: {server.url}", flush=True)
    print(f"Size: {server.size} bytes | Throttle: {rate} | Chunk: {server.chunk_size} B", flush=True)
    print(f"Layers: {', '.join(server.info['layers'])}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping RangeServer...", flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
