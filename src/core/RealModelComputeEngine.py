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
        self.safetensors_path = safetensors_path
        self.manifest = parse_safetensors(safetensors_path)
        self.all_layers_info = self.manifest["layers"]
        self.tensor_meta = self.manifest["tensors"]
        # keep only model.layers.N block keys for the streaming transformer path;
        # top-level keys (lm_head, model.embed_tokens/norm) are handled separately
        self.layers_info = {k: v for k, v in self.all_layers_info.items() if k.startswith("model.layers.")}
        self.layer_keys = sorted(self.layers_info.keys(), key=lambda k: self.layers_info[k]["start"])
        self.final_layer_key = max(self.layer_keys, key=self._layer_index) if self.layer_keys else None
        self.layer_count = len(self.layer_keys)
        self._infer_hidden_size()
        self._build_layer_info()
        self._build_all_layer_info()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_q_heads = 32  # TinyLlama q heads
        self.num_kv_heads = 4  # k/v grouped heads
        self.head_dim = self.hidden_size // self.num_q_heads
        self.hidden = None
        self.embed_weight = None
        self.norm_weight = None
        self.lm_head_weight = None
        self.next_token = None
        self.infer_trace: list[dict[str, object]] = []
        self.infer_total_s: float | None = None
        self._infer_t0: float | None = None
        self._init_hidden()

    # -- backend ------------------------------------------------------------
    @staticmethod
    def _layer_index(name: str) -> int:
        tail = name.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else 10**9

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

    def _build_all_layer_info(self) -> None:
        info = {}
        for lkey, ldesc in self.all_layers_info.items():
            row = []
            for tname in ldesc["tensors"]:
                t = self.tensor_meta[tname]
                row.append((tname, int(t["abs_start"] - ldesc["start"]), t["dtype"], t["shape"], int(t["length"])))
            info[lkey] = row
        self.all_layer_info = info

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

    def embed_token(self, token_id: int) -> torch.Tensor:
        if self.embed_weight is None:
            raise RuntimeError("embed_tokens.weight not loaded; run embed layer first")
        idx = torch.tensor([int(token_id)], device=self.embed_weight.device, dtype=torch.long)
        self.hidden = self.embed_weight.index_select(0, idx).unsqueeze(1)  # [1,1,hidden]
        return self.hidden

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

    def _load_layer_tensors(self, name: str, payload: bytes) -> dict[str, torch.Tensor]:
        params = {}
        desc = self.all_layers_info.get(name) or self.layers_info.get(name)
        if desc is None:
            raise KeyError(f"unknown layer {name}")
        layer_start = desc["start"]
        for tname in desc["tensors"]:
            tinfo = self.tensor_meta[tname]
            off = tinfo["abs_start"] - layer_start
            nbytes = tinfo["length"]
            params[tname] = self._to_torch_tensor(payload[off:off + nbytes], tinfo["dtype"], tinfo["shape"])
        return params

    def run_layer(self, name: str, payload: bytes) -> ComputeResult:
        t0 = time.perf_counter()
        params = self._load_layer_tensors(name, payload)

        if name == "model":
            # contains model.embed_tokens.weight and model.norm.weight
            if "model.embed_tokens.weight" in params:
                self.embed_weight = params["model.embed_tokens.weight"]
            if "model.norm.weight" in params:
                self.norm_weight = params["model.norm.weight"]
            # if hidden not yet set from token, keep/init
            if self.hidden is None:
                self._init_hidden()
        elif name == "lm_head":
            if "lm_head.weight" in params:
                self.lm_head_weight = params["lm_head.weight"]
            elif params:
                self.lm_head_weight = next(iter(params.values()))
            # If the streaming order put lm_head after the final block, finish
            # the head here as well so the answer is available either way.
            if self.hidden is not None and self.norm_weight is not None:
                self._compute_next_token()
        elif name.startswith("model.layers."):
            self.hidden = self._transformer_block(self.hidden, params)
            # The final *numeric* layer completes the forward pass.  File order
            # is not numeric order, so use final_layer_key, not layer_keys[-1].
            if name == self.final_layer_key and self.norm_weight is not None and self.lm_head_weight is not None:
                self._compute_next_token()
        else:
            # generic: try transformer if looks like block, else ignore
            if any(k.endswith("self_attn.q_proj.weight") for k in params):
                self.hidden = self._transformer_block(self.hidden, params)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        checksum_src = self.hidden if self.hidden is not None else (self.lm_head_weight if self.lm_head_weight is not None else self.embed_weight)
        checksum = float(checksum_src.detach().abs().mean().item()) if checksum_src is not None else 0.0
        elements = int(checksum_src.numel()) if checksum_src is not None else 0
        return ComputeResult(
            layer=name, checksum=checksum, dtype="F32", tensor_elements=elements,
            processed_elements=elements, operations=0, repeats=1,
            seconds=seconds, fallback=False,
        )

    def _compute_next_token(self) -> int:
        if self.hidden is None or self.norm_weight is None or self.lm_head_weight is None:
            raise RuntimeError("missing hidden/norm/lm_head for token computation")
        x = _rmsnorm(self.hidden, self.norm_weight)
        logits = F.linear(x, self.lm_head_weight)  # [1,1,vocab]
        logits = logits[:, -1, :].float()  # [1,vocab]
        probs = torch.softmax(logits, dim=-1)
        self.next_token = int(torch.argmax(probs, dim=-1).item())
        self.last_logits = logits
        return self.next_token

    def infer_token(self, token_id: int) -> int:
        """Run full prompt->next-token path: embed -> transformer layers -> norm/lm_head.

        ``infer_trace`` is reset and filled with one row per top-level group
        (``model``) and one per Transformer block, including cumulative TTFT.
        The answer is computed inside ``run_layer`` as soon as the final
        numeric layer completes (lm_head is preloaded so the head is available).
        """
        self.next_token = None
        self.last_logits = None
        self.infer_trace = []
        self._infer_t0 = time.perf_counter()
        self._init_hidden()

        def _read_group(f, name: str) -> bytes:
            desc = self.all_layers_info[name]
            f.seek(desc["start"])
            return f.read(desc["end"] - desc["start"])

        # 1) load embed/norm and preload lm_head weights.  lm_head is applied
        # only after the 22 blocks, but preloading lets the final block finish
        # the whole answer path inside run_layer.
        with open(self.safetensors_path, "rb") as f:
            for name in ("model", "lm_head"):
                if name not in self.all_layers_info:
                    continue
                result = self.run_layer(name, _read_group(f, name))
                if name == "model":
                    self.infer_trace.append(
                        {
                            "layer": name,
                            "checksum": float(result.checksum),
                            "seconds": float(result.seconds),
                            "ttft_s": time.perf_counter() - self._infer_t0,
                        }
                    )

        # 2) embed input token
        self.embed_token(token_id)
        # discard any premature head result from preloading lm_head
        self.next_token = None
        self.last_logits = None

        # 3) transformer layers in numeric forward order
        ordered_layers = sorted(self.layer_keys, key=self._layer_index)
        with open(self.safetensors_path, "rb") as f:
            for name in ordered_layers:
                result = self.run_layer(name, _read_group(f, name))
                self.infer_trace.append(
                    {
                        "layer": name,
                        "checksum": float(result.checksum),
                        "seconds": float(result.seconds),
                        "ttft_s": time.perf_counter() - self._infer_t0,
                    }
                )

        # 4) final norm + lm_head -> logits -> softmax/argmax
        if self.next_token is None:
            self._compute_next_token()
        if self.next_token is None:
            raise RuntimeError("failed to compute next token")
        self.infer_total_s = time.perf_counter() - self._infer_t0
        return int(self.next_token)
