#!/usr/bin/env python3
"""Prompt -> next-token answer path using the real engine (TinyLlama weights).

Builds the ``parse_safetensors`` manifest, creates ``RealModelComputeEngine``
(auto-selects cuda if available, else cpu), tokenises a short prompt, uses the
last token id as the start, calls ``engine.infer_token(start_id)``, and reports
the next-token index, per-layer TTFT, total wall time and per-layer checksums.

No tokenizer library is required.  If ``transformers``/``tokenizers`` and a
local tokenizer are importable they are used; otherwise a deterministic
byte-level fallback maps the prompt into the trained embedding rows (TinyLlama
reserves low ids for special/byte-fallback tokens, whose embeddings are ~0).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.safetensors_ranges import parse_safetensors  # noqa: E402
from src.core.RealModelComputeEngine import RealModelComputeEngine  # noqa: E402

FALLBACK_TOKEN_OFFSET = 1000  # avoid TinyLlama's ~zero special/byte rows


def tokenize_prompt(prompt: str, vocab_size: int) -> tuple[list[int], object | None, str]:
    """Return (token_ids, tokenizer_or_none, mode). No hard tokenizer dependency."""
    tokenizer = None
    # Try common tokenizer libraries if importable and a local tokenizer exists.
    for modname in ("transformers", "tokenizers"):
        try:
            if modname == "transformers":
                from transformers import AutoTokenizer  # type: ignore

                for cand in (ROOT / "models" / "tinyllama", ROOT / "models"):
                    try:
                        if cand.exists():
                            tokenizer = AutoTokenizer.from_pretrained(
                                str(cand), local_files_only=True
                            )
                            break
                    except Exception:
                        tokenizer = None
                if tokenizer is None:
                    raise RuntimeError("no local transformers tokenizer")
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                return [int(i) for i in ids], tokenizer, "transformers"
            else:
                from tokenizers import Tokenizer  # type: ignore

                for cand in (ROOT / "models" / "tokenizer.json", ROOT / "tokenizer.json"):
                    if cand.exists():
                        tokenizer = Tokenizer.from_file(str(cand))
                        enc = tokenizer.encode(prompt)
                        return [int(i) for i in enc.ids], tokenizer, "tokenizers"
                raise RuntimeError("no tokenizer.json")
        except Exception:
            tokenizer = None
            continue

    # Deterministic fallback: map UTF-8 bytes into the trained embedding range.
    offset = FALLBACK_TOKEN_OFFSET if vocab_size > FALLBACK_TOKEN_OFFSET + 1 else 0
    span = max(1, vocab_size - offset)
    ids = [offset + (b % span) for b in prompt.encode("utf-8")]
    if not ids:
        ids = [offset]
    return ids, None, f"bytes-fallback(offset={offset})"


def decode_token(token_id: int, tokenizer: object | None) -> str | None:
    if tokenizer is None:
        return None
    try:
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode([int(token_id)], skip_special_tokens=False)  # type: ignore[attr-defined]
        if hasattr(tokenizer, "id_to_token"):
            return tokenizer.id_to_token(int(token_id))  # type: ignore[attr-defined]
    except Exception:
        return None
    return None


def main() -> int:
    model_path = ROOT / "models" / "tinyllama.safetensors"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    manifest = parse_safetensors(str(model_path))
    engine = RealModelComputeEngine(str(model_path))  # auto-selects cuda if available
    embed_shape = manifest["tensors"]["model.embed_tokens.weight"]["shape"]
    vocab_size = int(embed_shape[0])

    prompt = "Hello"
    token_ids, tokenizer, tok_mode = tokenize_prompt(prompt, vocab_size)
    start_id = int(token_ids[-1])

    t0 = time.perf_counter()
    next_token = int(engine.infer_token(start_id))
    total_wall_s = time.perf_counter() - t0

    token_text = decode_token(next_token, tokenizer)
    trace = list(engine.infer_trace)
    layer_rows = [row for row in trace if str(row["layer"]).startswith("model.layers.")]
    checksums = [float(row["checksum"]) for row in layer_rows]

    print("Prompt -> Answer (real engine)")
    print(f"model_path      : {model_path}")
    print(f"device          : {engine.device}")
    print(f"prompt          : {prompt!r}")
    print(f"tokenizer_mode  : {tok_mode}")
    print(f"token_ids       : {token_ids}")
    print(f"start_token_id  : {start_id}")
    print(f"next_token_id   : {next_token}")
    if token_text is not None:
        print(f"next_token_text : {token_text!r}")
    else:
        print(f"next_token_text : {next_token}  # raw index (no tokenizer library)")
    print("answer_source   : model.embed_tokens.weight -> 22 layers -> model.norm.weight -> lm_head.weight -> softmax -> argmax")
    print(f"lm_head_shape   : {tuple(engine.lm_head_weight.shape)}")

    print()
    print("Per-layer metrics")
    print(f"{'layer':<20} {'ttft_s':>12} {'compute_s':>12} {'checksum':>16}")
    for row in trace:
        print(
            f"{str(row['layer']):<20} {float(row['ttft_s']):>12.6f} "
            f"{float(row['seconds']):>12.6f} {float(row['checksum']):>16.6f}"
        )

    finite_ok = bool(checksums) and all(math.isfinite(c) for c in checksums)
    rise_steps = sum(1 for a, b in zip(checksums, checksums[1:]) if b > a)
    rising_ok = len(checksums) >= 2 and checksums[-1] > checksums[0]
    layers_ok = len(layer_rows) == 22
    ok = finite_ok and rising_ok and layers_ok

    print()
    print(f"checksums_finite : {finite_ok}")
    print(f"checksums_rising : {rising_ok}")
    print(f"rising_definition: last transformer-layer checksum > first; "
          f"{rise_steps}/{max(0, len(checksums) - 1)} steps increase")
    print(f"transformer_layer_count : {len(layer_rows)}")
    print(f"checksum_count   : {len(checksums)}")
    print(f"total_wall_s     : {total_wall_s:.6f}")
    print(f"engine_total_s   : {float(engine.infer_total_s or 0.0):.6f}")
    print(f"exit_code        : {0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
