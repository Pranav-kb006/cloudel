#!/usr/bin/env python3
"""Generate a deterministic synthetic 2-layer safetensors model.

STATUS:
  BUILT: writes a valid safetensors container (8-byte LE header length, padded
         JSON header, raw F32 tensor data) plus a sidecar JSON range manifest.
         The harness/RangeServer can parse it with the stdlib parser, and
         ``src/core/SafetensorsManifest`` can parse it once core lands.
  TBD:   none.  This is a synthetic stand-in for a real transformer checkpoint;
         no model is trained and no accuracy claim is made.
  PACKAGES: standard library only.

Usage:
    python3 -m src.bench.make_synthetic_model \
        --out models/synthetic_2layer.safetensors --layers 2 --size-mb 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys

try:  # package import
    from ..storage.safetensors_ranges import parse_safetensors
except ImportError:  # direct execution / flat import
    from storage.safetensors_ranges import parse_safetensors  # type: ignore


def build_model(
    out_path: str,
    layers: int = 2,
    size_mb: float = 8.0,
    seed: int = 1234,
    cols: int = 512,
    write_manifest: bool = True,
) -> dict:
    """Create ``out_path`` and return the parsed range manifest."""
    if layers < 1:
        raise ValueError("--layers must be >= 1")
    if size_mb <= 0:
        raise ValueError("--size-mb must be > 0")
    if cols < 8:
        raise ValueError("--cols must be >= 8")

    target_bytes = int(size_mb * 1024 * 1024)
    per_layer = max(target_bytes // layers, cols * 4 * 2)
    bias_bytes = cols * 4
    weight_bytes = per_layer - bias_bytes
    rows = max(1, weight_bytes // (cols * 4))
    weight_bytes = rows * cols * 4
    layer_bytes = weight_bytes + bias_bytes

    rng = random.Random(seed)
    header: dict[str, dict] = {}
    data_parts: list[bytes] = []
    offset = 0

    for i in range(layers):
        wname = f"layer{i}.weight"
        bname = f"layer{i}.bias"
        header[wname] = {
            "dtype": "F32",
            "shape": [rows, cols],
            "data_offsets": [offset, offset + weight_bytes],
        }
        data_parts.append(rng.randbytes(weight_bytes))
        offset += weight_bytes

        header[bname] = {
            "dtype": "F32",
            "shape": [cols],
            "data_offsets": [offset, offset + bias_bytes],
        }
        data_parts.append(rng.randbytes(bias_bytes))
        offset += bias_bytes

    # safetensors requires the JSON header length to be a multiple of 8.
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (-len(header_json)) % 8
    header_json += b" " * pad

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        for part in data_parts:
            fh.write(part)
    os.replace(tmp_path, out_path)

    manifest = parse_safetensors(out_path)
    manifest["model"] = os.path.basename(out_path)
    manifest["synthetic"] = {
        "layers": layers,
        "rows": rows,
        "cols": cols,
        "seed": seed,
        "note": "synthetic F32 tensors; not a trained model",
    }

    if write_manifest:
        manifest_path = out_path + ".manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        manifest["manifest_path"] = manifest_path

    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="models/synthetic_2layer.safetensors")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--size-mb", type=float, default=8.0, help="approximate total size")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cols", type=int, default=512)
    p.add_argument("--no-manifest", action="store_true", help="skip sidecar JSON manifest")
    args = p.parse_args(argv)

    manifest = build_model(
        args.out,
        layers=args.layers,
        size_mb=args.size_mb,
        seed=args.seed,
        cols=args.cols,
        write_manifest=not args.no_manifest,
    )
    size_mb = manifest["size"] / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.2f} MiB, {manifest['size']} bytes)")
    print(f"Layers ({len(manifest['layers'])}):")
    for name, meta in manifest["layers"].items():
        print(f"  {name:<12} bytes [{meta['start']}:{meta['end']}] length={meta['length']}")
    if manifest.get("manifest_path"):
        print(f"Sidecar manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"make_synthetic_model failed: {type(exc).__name__}: {exc}\n")
        raise SystemExit(1)
