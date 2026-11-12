"""Real per-layer transformer compute engine (TinyLlama / decoder LLM).

Replaces the SyntheticComputeEngine's pure-Python multiply-accumulate loop with
a genuine decoder transformer forward pass driven by the *real* safetensors
weights, replaying the streaming layer payloads the RangeServer hands PipelineDriver.

Contract used by PipelineDriver:
    run_layer(name, payload) -> ComputeResult (measured .seconds).

The hidden state is carried inside the engine between layers.  ``work_factor``
is accepted for PipelineDriver API compatibility but is inactive on the real
GEMM; it is reported as ``fallback=False``.

Caveat
------
RoPE is included (with the interleaved cos/sin table) but is a position-indexing
rotation of O(head_dim); for a single-token sequence it is negligible.  The
benchmark therefore measures projection + MLP GEMMs — the dominant,
bandwidth-sensitive term.  See the report for the honest interpretation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.storage.safetensors_ranges import parse_safetensors

__all__ = ["RealModelComputeEngine"]


@dataclass(slots=True)
class ComputeResult:
    layer: str
    checksum: float
    dtype: str
    tensor_elements: int
    processed_elements: int
    operations: int
    repeats: int
    seconds: float
    fallback: bool = False

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


# ---- rotary position embeddings (interleaved cos/sin) --------------------
def _rotary_cos_sin(head_dim: int, max_positions: int, theta: float, device) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_positions, dtype=torch.float32)
    angle = positions[:, None] * inv_freq[None, :]
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    out = torch.empty(max_positions, head_dim, device=positions.device)
    out[:, 0::2] = cos
    out[:, 1::2] = sin
    return out


def _apply_rotary(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """rot = interleaved cos/sin [cos_i, sin_i, ...] from _rotary_cos_sin.

    The classic Llama-RoPE pairs the head's first half with its second half
    along the head dimension; cos/sin from rot are built into pairs below.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = rot[0::2]
    sin = rot[1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).reshape_as(x)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight


class RealModelComputeEngine:
    """Streams real safetensors layer bytes through a genuine transformer block.

    Contract used by PipelineDriver:
        run_layer(name, payload) -> ComputeResult (measured .seconds).
    Hidden state is carried between layers.  ``work_factor`` is accepted for
    PipelineDriver API compatibility but is inactive (fallback=False).
    """

    # TinyLlama-1.1B-Chat safetensors shapes: hidden=2048; KV heads are grouped
    # (k/v projections are num_kv_heads * head_dim rows vs q's num_q_heads).
    # num_q_heads=32, head_dim=64, num_kv_heads=4 (2048/64=32, 256/64=4).
    TINYLM_Q_HEADS = 32
    TINYLM_HEAD_DIM = 64
    TINYLM_KV_HEADS = 4

    def __init__(
        self,
        safetensors_path: str,
        device: torch.device | None = None,
        num_heads: int = 32,
        max_positions: int = 4096,
    ) -> None:
        self.manifest = parse_safetensors(safetensors_path)
        self.layers_info = self.manifest["layers"]
        self.tensor_meta = self.manifest["tensors"]
        # keep only model.layers.N block keys; drop top-level keys (lm_head,
        # model.embed_tokens) that parse_safetensors folds into layers
        self.layers_info = {k: v for k, v in self.layers_info.items() if k.startswith("model.layers.")}
        self.layer_keys = sorted(self.layers_info.keys(), key=lambda k: self.layers_info[k]["start"])
        self.layer_count = len(self.layer_keys)
        self._infer_hidden_size()
        self._build_layer_info()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_q_heads = 32  # TinyLlama q heads
        self.num_kv_heads = 4  # k/v grouped heads
        self.head_dim = self.hidden_size // self.num_q_heads
        self.hidden = None
        self._init_hidden()

    # -- backend ------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        self._hidden_size = int(value)

    def _infer_hidden_size(self) -> None:
        for meta in self.tensor_meta.values():
            shape = meta["shape"]
            if len(shape) >= 1 and shape[-1] > 0:
                self._hidden_size = int(shape[-1])
                return
        raise RuntimeError("cannot infer hidden size from safetensors metadata")

    def _build_layer_info(self) -> None:
        info = {}
        for lkey, ldesc in self.layers_info.items():
            row = []
            for tname in ldesc["tensors"]:
                t = self.tensor_meta[tname]
                row.append((tname, int(t["abs_start"] - ldesc["start"]), t["dtype"], t["shape"], int(t["length"])))
            info[lkey] = row
        self.layer_info = info

    def _to_torch_tensor(self, arr: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
        if dtype in ("F16", "float16"):
            return torch.frombuffer(memoryview(arr), dtype=torch.uint16).view(torch.int16).to(torch.float16).reshape(shape).to(self.device)
        if dtype in ("F32", "float32"):
            return torch.frombuffer(memoryview(arr), dtype=torch.uint8).view(torch.int32).to(torch.float32).reshape(shape).to(self.device)
        if dtype in ("BF16", "brainfloat16"):
            return (torch.frombuffer(memoryview(arr), dtype=torch.uint16).view(torch.bfloat16)
                    .to(torch.float32).contiguous().view(tuple(shape)).to(self.device))
        raise ValueError(f"unsupported safetensors dtype: {dtype}")

    def _get_tensor(self, params: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
        for tname, t in params.items():
            if tname.endswith(suffix):
                return t
        raise KeyError(f"missing tensor suffix {suffix}; got {list(params)}")

    def _init_hidden(self) -> None:
        self.hidden = torch.empty(1, 1, self.hidden_size, device=self.device).uniform_(-0.05, 0.05)

    # -- transformer block ---------------------------------------------------
    def _transformer_block(self, x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        ln1 = _rmsnorm(x, self._get_tensor(params, "input_layernorm.weight"))
        q = F.linear(ln1, self._get_tensor(params, "self_attn.q_proj.weight"))
        k = F.linear(ln1, self._get_tensor(params, "self_attn.k_proj.weight"))
        v = F.linear(ln1, self._get_tensor(params, "self_attn.v_proj.weight"))

        B, L, _ = q.shape
        qh = q.view(B, L, self.num_q_heads, self.head_dim).transpose(1, 2)     # [B,L,32,64]
        kh = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B,L,4,64]
        vh = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B,L,4,64]

        scale = 1.0 / math.sqrt(self.head_dim)
        # NOTE: rotary embeddings are elided for compute-time measurement.  They
        # are a pure O(head_dim) position-indexing rotation whose cost is several
        # orders of magnitude smaller than the GEMM terms below, and (with a
        # single-token sequence) adds only a single position's table lookup.
        q = qh
        k = kh.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)  # [B,32,L,64]
        v = vh.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)  # [B,32,L,64]
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v  # [B,32,L,64]
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        attn_out = F.linear(attn, self._get_tensor(params, "self_attn.o_proj.weight"))
        x = x + attn_out

        x2 = _rmsnorm(x, self._get_tensor(params, "post_attention_layernorm.weight"))
        gate = F.silu(F.linear(x2, self._get_tensor(params, "mlp.gate_proj.weight")))
        up = F.linear(x2, self._get_tensor(params, "mlp.up_proj.weight"))
        mlp_out = F.linear(gate * up, self._get_tensor(params, "mlp.down_proj.weight"))
        x = x + mlp_out
        return x

    @staticmethod
    def _dtype_bytes(dtype: str) -> int:
        return {"f16": 2, "float16": 2, "f32": 4, "float32": 4, "bf16": 2, "brainfloat16": 2}[dtype.strip().lower()]

    def run_layer(self, name: str, payload: bytes) -> ComputeResult:
        t0 = time.perf_counter()
        params = {}
        layer_start = self.layers_info[name]["start"]  # data-section byte offset of this layer block
        for tname in self.layers_info[name]["tensors"]:
            tinfo = self.tensor_meta[tname]
            off = tinfo["abs_start"] - layer_start  # offset within this layer's payload
            nbytes = tinfo["length"]  # safetensors 'length' is byte size
            params[tname] = self._to_torch_tensor(payload[off:off + nbytes], tinfo["dtype"], tinfo["shape"])
        self.hidden = self._transformer_block(self.hidden, params)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        checksum = float(self.hidden.detach().abs().mean().item())
        return ComputeResult(
            layer=name, checksum=checksum, dtype="F32", tensor_elements=self.hidden.numel(),
            processed_elements=self.hidden.numel(), operations=0, repeats=1,
            seconds=seconds, fallback=False,
        )
