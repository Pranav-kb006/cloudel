# Real-model end-to-end benchmark — TinyLlama-1.1B-Chat (measured)

Run: `venv/bin/python3 real_benchmark.py` → exit 0.
Data: `results/real_model_end_to_end.csv` (22 pipeline per-layer rows + 22 baseline
per-layer rows + summary rows).

**Runtime truth:** `torch 2.15.0+cpu`, `torch.cuda.is_available()=False`, engine
`device=cpu`. This run is **CPU-only**; no CUDA was active. The engine will select
CUDA when constructed with `device='cuda'` (or by its auto default) on a machine
where CUDA is available.

Model: `models/tinyllama.safetensors`, 2,200,119,864 bytes (2.098 GiB), 22
Transformer layers, hidden 2048, head_dim 64, kv_heads 4, BF16 weights.
Compute unit: one `seq=1` (single-token) forward pass per layer.

## Headline numbers

| metric | progressive (streaming) | baseline (full pre-download) |
|---|---|---|
| first-layer TTFT | **1.5120 s** | **30.4690 s** |
| full-pass TTFT / total | 28.0161 s | 36.0075 s |
| io_wait (blocked on I/O) | 25.5219 s | 30.3777 s |
| compute (22 real layers) | 2.4937 s | 2.1901 s |
| masking_ratio | **0.0000** | **0.0000** |
| bytes downloaded | 1,937,948,672 | 2,200,119,864 |
| peak RSS | 2408.3 MB | not sampled |

- First-layer TTFT is **20.1× lower** with progressive streaming (1.512 s vs
  30.469 s): layer 0 is fetched and computed without waiting for the full 2.2 GB.
- End-to-end is **7.99 s (22.2%) lower** (28.02 s vs 36.01 s).
- `masking_ratio = 1 - io_wait / nominal_io` clamps to **0** because the local
  RangeServer's measured throughput was ~579 Mbit/s vs the configured 1000
  Mbit/s nominal (per-chunk pacing overhead), so `io_wait > nominal_io`. Real
  per-layer compute (~0.09–0.29 s) is far too small to hide ~1.2 s/layer of
  transfer at this rate; the streaming win here is TTFT, not masking.

## Real per-layer compute_s (progressive run, CPU)

| layer | bytes | fetch_s | blocked_s | compute_s | checksum |
|---|---|---|---|---|---|
| model.layers.0 | 88088576 | 1.227168 | 1.226029 | 0.285991 | 0.025716 |
| model.layers.1 | 88088576 | 1.483314 | 1.197551 | 0.114993 | 0.026591 |
| model.layers.2 | 88088576 | 1.281070 | 1.166203 | 0.114907 | 0.031249 |
| model.layers.3 | 88088576 | 1.247172 | 1.132365 | 0.093798 | 0.032752 |
| model.layers.4 | 88088576 | 1.260837 | 1.167143 | 0.121295 | 0.036594 |
| model.layers.5 | 88088576 | 1.309528 | 1.188356 | 0.104312 | 0.040257 |
| model.layers.6 | 88088576 | 1.304062 | 1.199829 | 0.089987 | 0.042935 |
| model.layers.7 | 88088576 | 1.249955 | 1.160072 | 0.108558 | 0.062327 |
| model.layers.8 | 88088576 | 1.302786 | 1.194334 | 0.106102 | 0.072141 |
| model.layers.9 | 88088576 | 1.255736 | 1.149732 | 0.106932 | 0.076147 |
| model.layers.10 | 88088576 | 1.277877 | 1.171083 | 0.108139 | 0.076226 |
| model.layers.11 | 88088576 | 1.222572 | 1.114615 | 0.100124 | 0.080792 |
| model.layers.12 | 88088576 | 1.299288 | 1.199334 | 0.109802 | 0.078254 |
| model.layers.13 | 88088576 | 1.265445 | 1.155789 | 0.108994 | 0.080643 |
| model.layers.14 | 88088576 | 1.246728 | 1.138044 | 0.125820 | 0.087851 |
| model.layers.15 | 88088576 | 1.220042 | 1.094347 | 0.092240 | 0.112265 |
| model.layers.16 | 88088576 | 1.181882 | 1.089760 | 0.113812 | 0.128461 |
| model.layers.17 | 88088576 | 1.285576 | 1.172096 | 0.118485 | 0.158854 |
| model.layers.18 | 88088576 | 1.253620 | 1.135407 | 0.094578 | 0.194645 |
| model.layers.19 | 88088576 | 1.278700 | 1.184296 | 0.092420 | 0.260124 |
| model.layers.20 | 88088576 | 1.256999 | 1.164740 | 0.089597 | 0.375649 |
| model.layers.21 | 88088576 | 1.210227 | 1.120738 | 0.092844 | 0.668206 |

Compute mean **0.1134 s/layer**, median **0.1075 s**, total **2.4937 s** over 22
layers. Checksums are finite and rise overall (0.0257 → 0.6682), with small local
dips (layer 12 vs 11); hidden state is randomly initialized per run, so checksum
values are run-specific and are a sanity signal, not a model-quality metric.

## Honest scope

- CPU-only, single-token (`seq=1`) execution. Do not generalise to GPU or to
  production prompt lengths; with multi-token prompts the GEMM term grows and
  masking can rise.
- The pipeline fetches only the 22 layer blocks (1.94 GB), not `lm_head` /
  `embed_tokens`; the hidden state starts from the engine's random init rather
  than real token embeddings. Per-layer compute is nevertheless a genuine
  Transformer block forward pass over the real BF16 weights.
- Checkpoint storage order is lexicographic (0, 1, 10, …, 2, 20, …), so the
  benchmark reorders to numeric 0…21 before the forward pass.
- Baseline total includes ~3.44 s of copying layer slices out of the 2.2 GB
  in-memory body (`slice_s` in the CSV); that overhead is excluded from
  `compute_s`.
- Local RangeServer + HTTP loopback stands in for MinIO/S3 + `tc`/netem.
- `results/real_model_run.md` is a pre-fix standalone engine calibration (file
  order); this file and `results/real_model_end_to_end.csv` supersede it.
