"""Random-access parser for the ``safetensors`` file format.

The on-disk layout is::

    +--------------------------+  byte 0
    | uint64 little-endian N   |  8 bytes: header length
    +--------------------------+  byte 8
    | UTF-8 JSON header (N)    |  tensor table + optional __metadata__
    +--------------------------+  byte 8 + N
    | tensor data section      |  offsets in the header are relative here
    +--------------------------+

This module only touches the header when parsing a file (``mmap``), so a
multi-gigabyte checkpoint can be inspected without loading it.

STATUS:
  Built: TensorInfo/LayerRange dataclasses, header + metadata parsing from a
         path (mmap) or bytes, duplicate/bounds/alignment validation, natural
         layer grouping, and direct tensor/layer byte reads.
  TBD:   Nothing required for the EL artifact.  F8 variants are accepted as
         opaque byte tensors; no FP8 decoder is provided (compute engine is a
         synthetic FP32 stand-in).
  Packages: Python standard library only.  aiohttp/psutil are optional and not
         needed here; torch/numpy/safetensors are NOT required.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = [
    "SafetensorsError",
    "TensorInfo",
    "LayerRange",
    "SafetensorsManifest",
    "group_layer_name",
]

_HEADER_LEN = struct.Struct("<Q")
_U64 = struct.Struct("<Q")

# Itemsize in bytes for dtypes defined by the safetensors specification.
_DTYPE_ITEMSIZE: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

_LAYER_MARKERS = frozenset({"layers", "layer", "blocks", "block", "h"})
_NATURAL_SPLIT = re.compile(r"(\d+)")


class SafetensorsError(ValueError):
    """Raised when a safetensors header is malformed or out of bounds."""


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Byte range and metadata for one tensor.

    ``data_start``/``data_end`` are relative to the start of the data section,
    while ``abs_start``/``abs_end`` are absolute file offsets.  ``end`` values
    are exclusive, matching Python slicing and HTTP Range arithmetic.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    abs_start: int
    abs_end: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.data_end - self.data_start != self.nbytes:
            raise SafetensorsError(
                f"tensor {self.name!r}: relative range does not match nbytes"
            )
        if self.abs_end - self.abs_start != self.nbytes:
            raise SafetensorsError(
                f"tensor {self.name!r}: absolute range does not match nbytes"
            )

    @property
    def data_offsets(self) -> tuple[int, int]:
        """Return ``(data_start, data_end)`` as stored in the header."""
        return (self.data_start, self.data_end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_start": self.data_start,
            "data_end": self.data_end,
            "start": self.abs_start,
            "end": self.abs_end,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True, slots=True)
class LayerRange:
    """A group of tensors belonging to one top-level layer.

    ``start``/``end`` are absolute file offsets and span every tensor in the
    group, including any small alignment gaps between them.  ``length`` is
    ``end - start`` and is the number of bytes a prefetcher should request.
    """

    name: str
    tensors: tuple[TensorInfo, ...]
    start: int
    end: int
    length: int
    nbytes: int = 0
    dtypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise SafetensorsError(f"layer {self.name!r}: end < start")
        if self.length != self.end - self.start:
            raise SafetensorsError(f"layer {self.name!r}: length != end - start")
        if not self.nbytes:
            object.__setattr__(self, "nbytes", sum(t.nbytes for t in self.tensors))
        if not self.dtypes:
            object.__setattr__(
                self, "dtypes", tuple(sorted({t.dtype for t in self.tensors}))
            )

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tensors)

    @property
    def dtype(self) -> str | None:
        """Return the shared dtype, or ``None`` when the layer is mixed."""
        unique = set(self.dtypes)
        return next(iter(unique)) if len(unique) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "nbytes": self.nbytes,
            "dtypes": list(self.dtypes),
            "tensors": [t.to_dict() for t in self.tensors],
        }


def _natural_key(value: str) -> tuple[Any, ...]:
    """Sort ``model.layers.2`` before ``model.layers.10``.

    Numeric and text chunks are tagged so tuples never compare an ``int`` with
    a ``str`` (which would raise ``TypeError`` for names like ``1.weight``).
    """
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in _NATURAL_SPLIT.split(value)
        if part != ""
    )


def group_layer_name(tensor_name: str) -> str:
    """Map a tensor name to its streaming layer group.

    ``model.layers.3.self_attn.q_proj.weight`` -> ``model.layers.3``
    ``model.embed_tokens.weight``               -> ``model.embed_tokens``
    ``model.norm.weight``                       -> ``model.norm``
    ``lm_head.weight``                          -> ``lm_head``
    """
    parts = tensor_name.split(".")
    if len(parts) <= 1:
        return tensor_name
    for index, part in enumerate(parts[:-1]):
        if part in _LAYER_MARKERS and parts[index + 1].isdigit():
            return ".".join(parts[: index + 2])
    if parts[-1].isdigit() and len(parts) >= 2 and parts[-2] in _LAYER_MARKERS:
        return tensor_name
    return ".".join(parts[:-1])


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafetensorsError(f"duplicate JSON key in header: {key!r}")
        result[key] = value
    return result


class SafetensorsManifest:
    """Parsed tensor table and layer grouping for a safetensors file.

    Parameters
    ----------
    path_or_bytes:
        Filesystem path (``str``/``Path``) or an in-memory file image.
    max_header_bytes:
        Hard limit on the JSON header size (default 100 MiB) to bound work on
        hostile input.
    strict_header_alignment:
        Require the header length to be a multiple of 8, as the safetensors
        reference implementation does.
    check_tensor_alignment:
        Optionally require each tensor's absolute start to be aligned to its
        dtype itemsize.  Disabled by default because the format itself does
        not mandate it.
    """

    def __init__(
        self,
        path_or_bytes: str | os.PathLike[str] | bytes | bytearray | memoryview,
        *,
        max_header_bytes: int = 100 * 1024 * 1024,
        strict_header_alignment: bool = True,
        check_tensor_alignment: bool = False,
    ) -> None:
        if max_header_bytes < 8:
            raise ValueError("max_header_bytes must be >= 8")

        self.source: str | None = None
        self._raw: memoryview | None = None
        self.max_header_bytes = int(max_header_bytes)
        self.strict_header_alignment = bool(strict_header_alignment)
        self.check_tensor_alignment = bool(check_tensor_alignment)

        if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
            raw = memoryview(path_or_bytes)
            self._raw = raw
            self.file_size = len(raw)
            if self.file_size < 8:
                raise SafetensorsError(
                    f"file image is {self.file_size} bytes; need at least 8"
                )
            header_len = _U64.unpack(raw[:8])[0]
            self._validate_header_length(header_len)
            header_bytes = bytes(raw[8 : 8 + header_len])
            data_size = self.file_size - 8 - header_len
        else:
            path = os.fspath(path_or_bytes)
            if not isinstance(path, str):
                raise TypeError("path must be str or os.PathLike")
            self.source = path
            try:
                self.file_size = os.path.getsize(path)
            except OSError as exc:
                raise SafetensorsError(f"cannot stat {path!r}: {exc}") from exc
            if self.file_size < 8:
                raise SafetensorsError(
                    f"file {path!r} is {self.file_size} bytes; need at least 8"
                )
            try:
                with open(path, "rb") as handle:
                    mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        header_len = _U64.unpack(mm[:8])[0]
                        self._validate_header_length(header_len)
                        header_bytes = mm[8 : 8 + header_len]
                    finally:
                        mm.close()
            except OSError as exc:
                raise SafetensorsError(f"cannot read {path!r}: {exc}") from exc
            data_size = self.file_size - 8 - header_len

        self.header_len = int(header_len)
        self.data_offset = 8 + self.header_len
        self.data_size = int(data_size)
        self.header: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.layers: dict[str, LayerRange] = {}
        self._parse_header(header_bytes)
        self._build_layers()

    # ------------------------------------------------------------------ parse
    def _validate_header_length(self, header_len: int) -> None:
        if header_len < 2:
            raise SafetensorsError(f"header length {header_len} is too small")
        if header_len > self.max_header_bytes:
            raise SafetensorsError(
                f"header length {header_len} exceeds limit {self.max_header_bytes}"
            )
        if header_len > self.file_size - 8:
            raise SafetensorsError(
                f"header length {header_len} exceeds remaining file size "
                f"{self.file_size - 8}"
            )
        if self.strict_header_alignment and header_len % 8 != 0:
            raise SafetensorsError(
                f"header length {header_len} is not a multiple of 8"
            )

    def _parse_header(self, header_bytes: bytes) -> None:
        try:
            text = header_bytes.strip(b" \t\r\n\x00").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SafetensorsError(f"header is not valid UTF-8: {exc}") from exc
        if not text:
            raise SafetensorsError("header is empty")
        try:
            parsed = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except SafetensorsError:
            raise
        except json.JSONDecodeError as exc:
            raise SafetensorsError(f"header is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SafetensorsError("header JSON must be an object")
        self.header = parsed

        metadata = parsed.get("__metadata__", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise SafetensorsError("__metadata__ must be a JSON object")
        self.metadata = dict(metadata)

        for name, entry in parsed.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not name:
                raise SafetensorsError("tensor names must be non-empty strings")
            if not isinstance(entry, dict):
                raise SafetensorsError(f"tensor {name!r}: entry must be an object")
            self.tensors[name] = self._parse_tensor(name, entry)

    def _parse_tensor(self, name: str, entry: Mapping[str, Any]) -> TensorInfo:
        dtype = entry.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise SafetensorsError(f"tensor {name!r}: missing/invalid dtype")
        shape_raw = entry.get("shape")
        if not isinstance(shape_raw, (list, tuple)):
            raise SafetensorsError(f"tensor {name!r}: shape must be a list")
        shape: list[int] = []
        for dim in shape_raw:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
                raise SafetensorsError(
                    f"tensor {name!r}: shape dimensions must be non-negative ints"
                )
            shape.append(dim)
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            raise SafetensorsError(
                f"tensor {name!r}: data_offsets must be a two-element list"
            )
        data_start, data_end = offsets
        if (
            isinstance(data_start, bool)
            or isinstance(data_end, bool)
            or not isinstance(data_start, int)
            or not isinstance(data_end, int)
        ):
            raise SafetensorsError(
                f"tensor {name!r}: data_offsets must be integers"
            )
        if not (0 <= data_start <= data_end <= self.data_size):
            raise SafetensorsError(
                f"tensor {name!r}: data range [{data_start}, {data_end}) "
                f"outside data section of {self.data_size} bytes"
            )
        nbytes = data_end - data_start
        itemsize = _DTYPE_ITEMSIZE.get(dtype.upper())
        if itemsize is not None:
            expected = math.prod(shape) * itemsize
            if expected != nbytes:
                raise SafetensorsError(
                    f"tensor {name!r}: shape {tuple(shape)} x {dtype} needs "
                    f"{expected} bytes but data_offsets span {nbytes}"
                )
        abs_start = self.data_offset + data_start
        abs_end = self.data_offset + data_end
        if abs_end > self.file_size:
            raise SafetensorsError(
                f"tensor {name!r}: absolute end {abs_end} exceeds file size "
                f"{self.file_size}"
            )
        if self.check_tensor_alignment and itemsize:
            if abs_start % itemsize != 0:
                raise SafetensorsError(
                    f"tensor {name!r}: start {abs_start} not aligned to {itemsize}"
                )
        return TensorInfo(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            data_start=data_start,
            data_end=data_end,
            abs_start=abs_start,
            abs_end=abs_end,
            nbytes=nbytes,
        )

    def _build_layers(self) -> None:
        grouped: dict[str, list[TensorInfo]] = {}
        for tensor in self.tensors.values():
            grouped.setdefault(group_layer_name(tensor.name), []).append(tensor)
        layers: dict[str, LayerRange] = {}
        for layer_name in sorted(grouped, key=_natural_key):
            tensors = tuple(
                sorted(grouped[layer_name], key=lambda t: (t.abs_start, t.name))
            )
            start = min(t.abs_start for t in tensors)
            end = max(t.abs_end for t in tensors)
            layers[layer_name] = LayerRange(
                name=layer_name,
                tensors=tensors,
                start=start,
                end=end,
                length=end - start,
            )
        self.layers = layers

    # --------------------------------------------------------------- accessors
    @property
    def layer_names(self) -> list[str]:
        """Layer groups in deterministic natural order."""
        return list(self.layers)

    @property
    def tensor_names(self) -> list[str]:
        return list(self.tensors)

    @property
    def dtypes(self) -> list[str]:
        return sorted({t.dtype for t in self.tensors.values()})

    @property
    def total_nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def get_tensor(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"unknown tensor {name!r}") from None

    def get_layer(self, name: str) -> LayerRange:
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(f"unknown layer {name!r}") from None

    def read_tensor(self, name: str) -> bytes:
        """Return the exact stored bytes for ``name``.

        For path-backed manifests this opens the file and seeks directly to the
        tensor, so no whole-file copy is made.
        """
        tensor = self.get_tensor(name)
        if self._raw is not None:
            return bytes(self._raw[tensor.abs_start : tensor.abs_end])
        if self.source is None:  # pragma: no cover - defensive
            raise SafetensorsError("manifest has no backing storage")
        with open(self.source, "rb") as handle:
            handle.seek(tensor.abs_start)
            data = handle.read(tensor.nbytes)
        if len(data) != tensor.nbytes:
            raise SafetensorsError(
                f"short read for {name!r}: {len(data)} != {tensor.nbytes}"
            )
        return data

    def read_layer(self, name: str) -> bytes:
        """Return the contiguous byte span covering a layer group."""
        layer = self.get_layer(name)
        if self._raw is not None:
            return bytes(self._raw[layer.start : layer.end])
        if self.source is None:  # pragma: no cover - defensive
            raise SafetensorsError("manifest has no backing storage")
        with open(self.source, "rb") as handle:
            handle.seek(layer.start)
            data = handle.read(layer.length)
        if len(data) != layer.length:
            raise SafetensorsError(
                f"short read for layer {name!r}: {len(data)} != {layer.length}"
            )
        return data

    def verify_offsets(self) -> dict[str, int]:
        """Read every tensor back and confirm its byte count.

        Returns a mapping of tensor name -> bytes read.  Raises on mismatch.
        """
        checked: dict[str, int] = {}
        for name, tensor in self.tensors.items():
            data = self.read_tensor(name)
            if len(data) != tensor.nbytes:
                raise SafetensorsError(f"offset check failed for {name!r}")
            checked[name] = len(data)
        return checked

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": Path(self.source).name if self.source else None,
            "size": self.file_size,
            "header_len": self.header_len,
            "data_offset": self.data_offset,
            "data_size": self.data_size,
            "metadata": self.metadata,
            "tensors": {name: t.to_dict() for name, t in self.tensors.items()},
            "layers": {name: layer.to_dict() for name, layer in self.layers.items()},
        }

    def __len__(self) -> int:
        return len(self.tensors)

    def __contains__(self, name: object) -> bool:
        return name in self.tensors

    def __iter__(self) -> Iterator[TensorInfo]:
        return iter(self.tensors.values())

    def __repr__(self) -> str:
        return (
            f"SafetensorsManifest(tensors={len(self.tensors)}, "
            f"layers={len(self.layers)}, size={self.file_size})"
        )
