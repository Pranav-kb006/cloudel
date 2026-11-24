"""Generate a genuine Llama-architecture Transformer checkpoint in safetensors format.

Creates a valid, spec-compliant safetensors file with real BF16 decoder-transformer
weights (embed_tokens, N x Llama blocks with RMSNorm + GQA q/k/v/o_proj + SwiGLU
gate/up/down_proj, final norm, and lm_head), laid out in contiguous per-layer byte
ranges so HTTP Range streaming works with zero wasted bytes.

Default profile ("llama_12layer_140m"):
  - 12 decoder layers, hidden=1024, intermediate=2816, q_heads=16, kv_heads=4, vocab=8192
  - Per-layer weight size: ~22.5 MiB (BF16) -> total checkpoint ~286 MiB
  - Ideal for reproducible CPU/GPU serverless streaming benchmarks and live demos.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.safetensors_ranges import parse_safetensors


def build_transformer_safetensors(
    output_path: str | Path,
    *,
    num_layers: int = 12,
    hidden_size: int = 1024,
    intermediate_size: int = 2816,
    num_q_heads: int = 16,
    num_kv_heads: int = 4,
    vocab_size: int = 8192,
    seed: int = 2026,
) -> dict:
    """Write a Llama-style BF16 safetensors file and its range manifest."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    head_dim = hidden_size // num_q_heads
    kv_dim = num_kv_heads * head_dim
    gen = torch.Generator().manual_seed(seed)

    # Ordered list of (tensor_name, shape, dtype_str) grouped contiguously by layer
    specs: list[tuple[str, list[int], str]] = [
        ("model.embed_tokens.weight", [vocab_size, hidden_size], "BF16"),
        ("model.norm.weight", [hidden_size], "BF16"),
    ]
    for i in range(num_layers):
        p = f"model.layers.{i}"
        specs.extend([
            (f"{p}.input_layernorm.weight", [hidden_size], "BF16"),
            (f"{p}.self_attn.q_proj.weight", [hidden_size, hidden_size], "BF16"),
            (f"{p}.self_attn.k_proj.weight", [kv_dim, hidden_size], "BF16"),
            (f"{p}.self_attn.v_proj.weight", [kv_dim, hidden_size], "BF16"),
            (f"{p}.self_attn.o_proj.weight", [hidden_size, hidden_size], "BF16"),
            (f"{p}.post_attention_layernorm.weight", [hidden_size], "BF16"),
            (f"{p}.mlp.gate_proj.weight", [intermediate_size, hidden_size], "BF16"),
            (f"{p}.mlp.up_proj.weight", [intermediate_size, hidden_size], "BF16"),
            (f"{p}.mlp.down_proj.weight", [hidden_size, intermediate_size], "BF16"),
        ])
    specs.append(("lm_head.weight", [vocab_size, hidden_size], "BF16"))

    # Build safetensors header first
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "architecture": "llama_decoder",
            "num_layers": str(num_layers),
            "hidden_size": str(hidden_size),
            "intermediate_size": str(intermediate_size),
            "num_q_heads": str(num_q_heads),
            "num_kv_heads": str(num_kv_heads),
            "vocab_size": str(vocab_size),
        }
    }
    offset = 0
    for name, shape, dtype in specs:
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * 2  # BF16 = 2 bytes
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad

    with open(out, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for name, shape, _ in specs:
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                t = torch.ones(shape, dtype=torch.bfloat16)
            else:
                std = (hidden_size ** -0.5) * 0.5
                t = (torch.randn(shape, generator=gen, dtype=torch.float32) * std).to(torch.bfloat16)
            fh.write(t.view(torch.uint16).contiguous().numpy().tobytes())

    info = parse_safetensors(str(out))
    manifest_path = Path(str(out) + ".manifest.json")
    manifest_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    total_mb = os.path.getsize(out) / (1024 * 1024)
    layer0 = info["layers"]["model.layers.0"]
    layer_mb = (layer0["end"] - layer0["start"]) / (1024 * 1024)
    return {
        "path": str(out),
        "manifest": str(manifest_path),
        "total_mb": round(total_mb, 2),
        "num_layers": num_layers,
        "layer_mb": round(layer_mb, 2),
        "params_m": round(offset / 2 / 1e6, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Llama-architecture safetensors checkpoint.")
    ap.add_argument("--out", default=str(ROOT / "models" / "llama_12layer_140m.safetensors"))
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--intermediate", type=int, default=2816)
    ap.add_argument("--q-heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    res = build_transformer_safetensors(
        args.out,
        num_layers=args.layers,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        vocab_size=args.vocab,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
