"""S3 / MinIO HTTP Range adapter for disaggregated safetensors layer streaming.

Supports both:
  1. Direct HTTP Range GET against MinIO / AWS S3 (public bucket or presigned URL),
     fetching only the 8-byte safetensors header length + JSON tensor index first
     (two tiny Range requests) before streaming individual layer byte ranges.
  2. Automatic fallback to local RangeServer when MinIO is not running.
"""

from __future__ import annotations

import json
import os
import struct
import urllib.request
from typing import Any


def fetch_remote_safetensors_manifest(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """Read ONLY the safetensors header over HTTP Range (zero full-file download).

    Issues two small byte-range requests:
      - Range: bytes=0-7          -> 8-byte little-endian uint64 header length N
      - Range: bytes=8-(8+N-1)    -> JSON tensor metadata header
    Then computes per-layer contiguous byte ranges identical to parse_safetensors().
    """
    req_len = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req_len, timeout=timeout) as resp:
        raw_len = resp.read()
    if len(raw_len) != 8:
        raise RuntimeError(f"Expected 8 header-length bytes, got {len(raw_len)}")
    header_len = struct.unpack("<Q", raw_len)[0]

    req_hdr = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + header_len - 1}"})
    with urllib.request.urlopen(req_hdr, timeout=timeout) as resp:
        header_bytes = resp.read()
    header = json.loads(header_bytes.decode("utf-8"))

    data_base = 8 + header_len
    tensors: dict[str, dict[str, Any]] = {}
    layer_groups: dict[str, list[str]] = {}

    for tname, meta in header.items():
        if tname == "__metadata__":
            continue
        rel_start, rel_end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        abs_start, abs_end = data_base + rel_start, data_base + rel_end
        tensors[tname] = {
            "dtype": meta["dtype"],
            "shape": list(meta["shape"]),
            "abs_start": abs_start,
            "abs_end": abs_end,
            "length": abs_end - abs_start,
        }
        parts = tname.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
            lkey = f"model.layers.{parts[2]}"
        elif parts[0] == "lm_head":
            lkey = "lm_head"
        else:
            lkey = parts[0]
        layer_groups.setdefault(lkey, []).append(tname)

    layers: dict[str, dict[str, Any]] = {}
    for lkey, tnames in layer_groups.items():
        starts = [tensors[t]["abs_start"] for t in tnames]
        ends = [tensors[t]["abs_end"] for t in tnames]
        layers[lkey] = {
            "start": min(starts),
            "end": max(ends),
            "length": max(ends) - min(starts),
            "tensors": tnames,
        }

    return {
        "source_url": url,
        "header_len": header_len,
        "data_base": data_base,
        "tensors": tensors,
        "layers": layers,
    }


def check_minio_available(endpoint: str | None = None, timeout: float = 2.0) -> bool:
    """Return True if a MinIO S3 server is reachable at endpoint."""
    ep = (endpoint or os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000")).rstrip("/")
    try:
        req = urllib.request.Request(f"{ep}/minio/health/live", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False
