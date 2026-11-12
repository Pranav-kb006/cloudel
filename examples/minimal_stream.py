#!/usr/bin/env python3
"""End-to-end minimal demo of the async layer-streaming engine.

What it does
------------
1. Builds a tiny valid 2-transformer-layer safetensors file (FP32, no torch).
2. Serves it from a threaded stdlib HTTP server that honours Range requests.
3. Parses the manifest, writes a JSON chunk manifest, and checks that every
   tensor's absolute byte offsets reconstruct the original bytes.
4. Streams every layer group with AsyncPrefetcher + PipelineDriver, computes a
   deterministic synthetic matvec/elementwise kernel on the real bytes, and
   prints TTFT, per-layer download/compute times, and per-layer RSS.

Run from the repository root::

    python3 examples/minimal_stream.py

STATUS:
  Built: fully runnable stdlib demo (synthetic model, Range server, manifest,
         offset check, prefetch pipeline, metrics output).
  TBD:   Real checkpoint / MinIO / bandwidth shaping belong to WS-3's bench.
  Packages: Python standard library only.  aiohttp/psutil are optional;
            torch/numpy/safetensors are NOT required.
"""

from __future__ import annotations

import array
import asyncio
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import (  # noqa: E402  (path bootstrap must run first)
    AsyncPrefetcher,
    MemoryBuffer,
    PipelineDriver,
    S3ManifestWriter,
    SafetensorsManifest,
    SyntheticComputeEngine,
)

MODEL_ROUTE = "/model.safetensors"
# Loopback stand-in for storage/network latency and tunable synthetic compute.
# Defaults are chosen so the demo shows both a measurable TTFT and real overlap;
# override with MINIMAL_SERVER_DELAY_S / MINIMAL_WORK_FACTOR if desired.
SERVER_DELAY_S = float(os.environ.get("MINIMAL_SERVER_DELAY_S", "0.01"))
WORK_FACTOR = float(os.environ.get("MINIMAL_WORK_FACTOR", "6"))


# --------------------------------------------------------------------------- model
def _float_values(count: int, seed: int) -> list[float]:
    """Deterministic pseudo-random floats in [-0.5, 0.5)."""
    state = seed & ((1 << 64) - 1) or 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    values: list[float] = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        values.append((state >> 11) / float(1 << 53) - 0.5)
    return values


def _fp32_bytes(values: list[float]) -> bytes:
    payload = array.array("f", values)
    if sys.byteorder == "big":
        payload.byteswap()
    return payload.tobytes()


def build_synthetic_safetensors(
    path: Path,
) -> dict[str, bytes]:
    """Write a valid safetensors file and return its raw tensor bytes.

    The tensor table uses a 2-layer transformer-like naming scheme so the
    manifest grouping is exercised (layers 0/1, embeddings, norm, lm_head).
    """
    tensors: list[tuple[str, tuple[int, ...], int]] = [
        ("model.embed_tokens.weight", (32, 32), 11),
        ("model.layers.0.self_attn.q_proj.weight", (32, 32), 21),
        ("model.layers.0.mlp.up_proj.weight", (64, 32), 22),
        ("model.layers.1.self_attn.q_proj.weight", (32, 32), 31),
        ("model.layers.1.mlp.up_proj.weight", (64, 32), 32),
        ("model.norm.weight", (32,), 41),
        ("lm_head.weight", (32, 32), 51),
    ]

    header: dict[str, Any] = {
        "__metadata__": {"format": "pt", "producer": "minimal_stream.py"}
    }
    blobs: list[bytes] = []
    raw: dict[str, bytes] = {}
    offset = 0
    for name, shape, seed in tensors:
        count = 1
        for dimension in shape:
            count *= dimension
        payload = _fp32_bytes(_float_values(count, seed))
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        blobs.append(payload)
        raw[name] = payload
        offset += len(payload)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    assert len(header_bytes) % 8 == 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for payload in blobs:
            handle.write(payload)
    return raw


# -------------------------------------------------------------------------- server
class RangeRequestHandler(BaseHTTPRequestHandler):
    """Serve one file with single-range ``bytes=start-end`` support."""

    file_path: Path
    delay_s: float = 0.0
    _range_re = re.compile(r"^bytes=(\d*)-(\d*)$")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep the demo output clean; errors are still visible via responses.
        return

    def _resolve_range(self, total: int) -> tuple[int, int, bool] | None:
        header = self.headers.get("Range")
        if not header:
            return (0, total - 1, False)
        match = self._range_re.fullmatch(header.strip())
        if not match:
            return None
        start_text, end_text = match.groups()
        if start_text == "" and end_text == "":
            return None
        if start_text == "":
            length = int(end_text)
            if length <= 0:
                return None
            start = max(0, total - length)
            return (start, total - 1, True)
        start = int(start_text)
        end = int(end_text) if end_text else total - 1
        if start >= total or end < start:
            return None
        return (start, min(end, total - 1), True)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != MODEL_ROUTE:
            self.send_error(404, "not found")
            return
        data = self.file_path.read_bytes()
        total = len(data)
        resolved = self._resolve_range(total)
        if resolved is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()
            return
        start, end, partial = resolved
        body = data[start : end + 1]
        if self.delay_s:
            time.sleep(self.delay_s)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        self.wfile.write(body)


def start_server(model_path: Path) -> tuple[ThreadingHTTPServer, str]:
    handler = type(
        "MinimalRangeHandler",
        (RangeRequestHandler,),
        {"file_path": model_path, "delay_s": SERVER_DELAY_S},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}{MODEL_ROUTE}"


# --------------------------------------------------------------------------- demo
def verify_offsets(manifest: SafetensorsManifest, raw: dict[str, bytes]) -> None:
    """Assert that absolute manifest offsets reconstruct the original bytes."""
    if set(manifest.tensors) != set(raw):
        raise AssertionError("manifest tensor names do not match source")
    for name, expected in raw.items():
        actual = manifest.read_tensor(name)
        if actual != expected:
            raise AssertionError(f"byte mismatch for tensor {name!r}")
        info = manifest.tensors[name]
        if info.abs_end - info.abs_start != len(expected):
            raise AssertionError(f"range length mismatch for {name!r}")


async def run_pipeline(url: str, manifest: SafetensorsManifest) -> dict[str, Any]:
    buffer = MemoryBuffer(max_depth=3, rss_limit_mb=2048)
    prefetcher = AsyncPrefetcher(
        url,
        manifest,
        buffer=buffer,
        lookahead=2,
        timeout=10.0,
        retries=2,
    )
    engine = SyntheticComputeEngine(work_factor=WORK_FACTOR, hidden_dim=32)
    driver = PipelineDriver(prefetcher, compute_engine=engine, buffer=buffer)

    await prefetcher.start()
    try:
        metrics = await driver.run()
    finally:
        await prefetcher.stop()
    return metrics


def print_report(metrics: dict[str, Any], chunk: dict[str, Any]) -> None:
    print()
    print("Per-layer streaming report")
    print("-" * 92)
    print(
        f"{'layer':<40} {'bytes':>8} {'dl_ms':>8} {'wait_ms':>9} "
        f"{'comp_ms':>9} {'rss_MB':>9}"
    )
    for row in metrics["per_layer"]:
        print(
            f"{row['layer']:<40} {row['bytes']:>8} "
            f"{row['download_s'] * 1000:>8.2f} "
            f"{row['io_wait_s'] * 1000:>9.2f} "
            f"{row['compute_s'] * 1000:>9.2f} "
            f"{row['rss_mb']:>9.2f}"
        )
    print("-" * 92)
    prefetch = metrics.get("prefetcher_stats", {})
    buffer_stats = metrics.get("buffer_stats", {})
    print(f"TTFT (full forward complete)   : {metrics['ttft_s'] * 1000:.2f} ms")
    print(f"First layer compute done        : {metrics.get('first_layer_s', 0.0) * 1000:.2f} ms")
    print(f"First payload arrival           : {metrics['first_payload_s'] * 1000:.2f} ms")
    print(f"Total stream time               : {metrics['total_s'] * 1000:.2f} ms")
    print(f"I/O wait total                  : {metrics['io_wait_s'] * 1000:.2f} ms")
    print(f"Compute total                   : {metrics['compute_s'] * 1000:.2f} ms")
    print(f"Masking ratio (compute/(c+i))   : {metrics['masking_ratio'] * 100:.1f}%")
    print(f"I/O masking ratio (1-wait/total): {metrics['io_masking_ratio'] * 100:.1f}%")
    print(f"Layers / bytes downloaded       : {metrics['layers']} / {metrics['bytes_downloaded']}")
    print(f"Prefetch backend                : {prefetch.get('backend')}")
    print(f"Lookahead (configured/effective): {prefetch.get('lookahead')} / {prefetch.get('effective_lookahead')}")
    print(f"Buffer evictions / drops        : {buffer_stats.get('evictions')} / {buffer_stats.get('drops')}")
    print(f"Chunk-manifest layers           : {len(chunk.get('layers', {}))}")


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="layer-stream-") as temp_dir:
        work = Path(temp_dir)
        model_path = work / "minimal_model.safetensors"
        raw = build_synthetic_safetensors(model_path)

        manifest = SafetensorsManifest(model_path)
        verify_offsets(manifest, raw)

        writer = S3ManifestWriter(
            manifest,
            "models/minimal/minimal_model.safetensors",
            bucket="local-dev",
            model_name="minimal-2layer",
        )
        chunk = writer.write(work / "chunk_manifest.json")

        server, url = start_server(model_path)
        try:
            print("Async progressive layer-streaming demo")
            print(f"model file      : {model_path}")
            print(f"model size      : {manifest.file_size} bytes")
            print(f"tensors / layers: {len(manifest.tensors)} / {len(manifest.layers)}")
            print(f"offset check    : {len(raw)}/{len(raw)} tensors match")
            print(f"chunk manifest  : {work / 'chunk_manifest.json'}")
            print(f"range server    : {url}")
            print(f"work factor     : {WORK_FACTOR}")
            print("layer groups    : " + ", ".join(manifest.layer_names))

            metrics = await run_pipeline(url, manifest)
            print_report(metrics, chunk)
        finally:
            server.shutdown()
            server.server_close()

    print()
    print("OK: async layer streaming completed end-to-end (stdlib only).")


if __name__ == "__main__":
    asyncio.run(main())
