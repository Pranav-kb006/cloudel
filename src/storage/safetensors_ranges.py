"""Minimal, dependency-free safetensors header/range parser.

STATUS:
  BUILT: parses the 8-byte little-endian header length, the JSON header, and
         exposes absolute byte offsets for every tensor and a grouping by
         top-level layer.  Used by RangeServer, the synthetic model generator
         and the benchmark harness.
  TBD:   nothing for the local artifact.  When ``src/core/SafetensorsManifest``
         is available the harness prefers it for the canonical manifest and
         uses this module only as a fallback.
  PACKAGES: standard library only.

SAFETENSORS LAYOUT (what this module reads):
    [0:8]            uint64 little-endian header length N
    [8:8+N]          UTF-8 JSON header (padded with spaces, N % 8 == 0)
    [8+N:...]        raw tensor data
  Each tensor entry is {"dtype": ..., "shape": [...],
                        "data_offsets": [rel_start, rel_end]}
  where offsets are relative to the start of the data section (8+N).
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any


class SafetensorsFormatError(ValueError):
    """Raised when a file is not a valid safetensors container."""


@dataclass
class TensorRange:
    name: str
    dtype: str
    shape: list[int]
    rel_start: int
    rel_end: int
    abs_start: int
    abs_end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "rel_start": self.rel_start,
            "rel_end": self.rel_end,
            "abs_start": self.abs_start,
            "abs_end": self.abs_end,
            "length": self.abs_end - self.abs_start,
        }


@dataclass
class LayerRange:
    name: str
    start: int
    end: int
    tensors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.end - self.start,
            "tensors": list(self.tensors),
        }


_LAYER_DOT = re.compile(r"^(.*?\.layers\.(\d+))")
_LAYER_UNDER = re.compile(r"^(.*?layers?[._-](\d+))", re.IGNORECASE)
_LAYER_SHORT = re.compile(r"^layer[._-]?(\d+)", re.IGNORECASE)


def layer_key(tensor_name: str) -> str:
    """Map a tensor name to a stable layer key.

    Examples
    --------
    ``layer0.weight``                    -> ``layer0``
    ``model.layers.3.self_attn.q.weight``-> ``model.layers.3``
    ``transformer.h.11.mlp.fc1.weight``  -> ``transformer.h.11`` (via layers? no)
    ``embed.weight``                     -> ``embed``
    """
    if tensor_name == "__metadata__":
        return "__metadata__"

    m = _LAYER_SHORT.match(tensor_name)
    if m:
        return f"layer{int(m.group(1))}"

    m = _LAYER_DOT.match(tensor_name)
    if m:
        return m.group(1)

    m = _LAYER_UNDER.match(tensor_name)
    if m:
        return m.group(1)

    # Fall back to the first path component so unrelated tensors stay grouped.
    return tensor_name.split(".", 1)[0] or "default"


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a safetensors header.

    Returns a dict with ``header`` (parsed JSON), ``header_len``,
    ``data_start``, ``size`` and ``tensors`` (list[TensorRange]).
    """
    path = os.fspath(path)
    size = os.path.getsize(path)
    if size < 9:
        raise SafetensorsFormatError(f"{path}: file too small ({size} bytes)")

    with open(path, "rb") as fh:
        raw = fh.read(8)
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0 or header_len > size - 8:
            raise SafetensorsFormatError(
                f"{path}: invalid header length {header_len} for file size {size}"
            )
        header_bytes = fh.read(header_len)
        if len(header_bytes) != header_len:
            raise SafetensorsFormatError(f"{path}: truncated JSON header")

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(f"{path}: invalid JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsFormatError(f"{path}: header is not a JSON object")

    data_start = 8 + header_len
    data_size = size - data_start
    tensors: list[TensorRange] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise SafetensorsFormatError(f"{path}: tensor {name!r} metadata is not an object")
        try:
            rel_start, rel_end = meta["data_offsets"]
            dtype = str(meta.get("dtype", "unknown"))
            shape = [int(x) for x in meta.get("shape", [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsFormatError(f"{path}: bad metadata for tensor {name!r}") from exc
        if not (0 <= rel_start <= rel_end <= data_size):
            raise SafetensorsFormatError(
                f"{path}: tensor {name!r} offsets [{rel_start},{rel_end}] "
                f"outside data section of {data_size} bytes"
            )
        tensors.append(
            TensorRange(
                name=name,
                dtype=dtype,
                shape=shape,
                rel_start=rel_start,
                rel_end=rel_end,
                abs_start=data_start + rel_start,
                abs_end=data_start + rel_end,
            )
        )

    if not tensors:
        raise SafetensorsFormatError(f"{path}: no tensors found")

    return {
        "header": header,
        "header_len": header_len,
        "data_start": data_start,
        "size": size,
        "tensors": tensors,
    }


def parse_safetensors(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse a safetensors file into a JSON-serialisable range manifest.

    The returned mapping is deliberately simple so it can be passed to the
    AsyncPrefetcher / S3ManifestWriter contract:

        {
          "file": path, "size": int, "header_len": int, "data_start": int,
          "tensors": {name: {...abs_start/abs_end...}},
          "layers":  {layer: {"start": int, "end": int,
                              "length": int, "tensors": [names]}}
        }
    """
    info = read_header(path)
    tensors: dict[str, dict[str, Any]] = {}
    layer_map: dict[str, LayerRange] = {}

    for t in info["tensors"]:
        tensors[t.name] = t.as_dict()
        key = layer_key(t.name)
        lr = layer_map.get(key)
        if lr is None:
            layer_map[key] = LayerRange(
                name=key, start=t.abs_start, end=t.abs_end, tensors=[t.name]
            )
        else:
            lr.start = min(lr.start, t.abs_start)
            lr.end = max(lr.end, t.abs_end)
            lr.tensors.append(t.name)

    # Stable ordering by file offset (the natural streaming order).
    ordered = sorted(layer_map.values(), key=lambda lr: (lr.start, lr.name))
    layers = {lr.name: lr.as_dict() for lr in ordered}

    return {
        "file": os.fspath(path),
        "size": info["size"],
        "header_len": info["header_len"],
        "data_start": info["data_start"],
        "tensors": tensors,
        "layers": layers,
    }
