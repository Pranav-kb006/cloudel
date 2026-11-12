# Real-model benchmark results (TinyLlama-1.1B-Chat via `RealModelComputeEngine`)

> Superseded by `results/README.md` and `results/real_model_end_to_end.csv`
> (end-to-end progressive-vs-baseline run in numeric layer order). Kept as the
> earlier standalone engine calibration.

Verified 2026-11-11.  Hardware: CPU (torch CPU wheel; CUDA not active).
Model: `models/tinyllama.safetensors` (2.05 GB > 22 Transformer layers, hidden 2048,
head_dim 64, grouped-query KV heads = 4).  Compute per layer = `seq=1` single-token
forward pass.

## Per-layer results

| layer | file offset bytes | compute_s | checksum (|h| mean) |
|---|---|---|---|
| model.layers.0 | 262,167,096 | 0.1759 | 0.02424 |
| model.layers.1 | 350,255,672 | 0.0965 | 0.02458 |
| model.layers.10 | 1,496,770,304 | 0.1141 | 0.07419 |
| model.layers.11 | 1,584,858,880 | 0.0867 | 0.10346 |
| model.layers.12 | 1,672,947,456 | 0.0908 | 0.10965 |
| model.layers.13 | 1,761,036,032 | 0.0920 | 0.11252 |
| model.layers.14 | 1,849,124,608 | 0.0878 | 0.11188 |
| model.layers.15 | 1,937,213,184 | 0.0992 | 0.12973 |
| model.layers.16 | 2,025,301,760 | 0.1013 | 0.14095 |
| model.layers.17 | 2,113,390,336 | 0.1129 | 0.16744 |
| model.layers.18 | 2,201,478,912 | 0.1147 | 0.19797 |
| model.layers.19 | 2,289,567,488 | 0.1002 | 0.25564 |
| model.layers.2 | 350,255,672 | 0.0969 | 0.2547 |
| model.layers.20 | 2,377,656,064 | 0.0903 | 0.34447 |
| model.layers.21 | 2,465,744,640 | 0.1010 | 0.53908 |
| model.layers.3 | 438,344,240 | 0.1120 | 0.54078 |
| model.layers.4 | 526,432,816 | 0.1010 | 0.54389 |
| model.layers.5 | 614,521,392 | 0.0929 | 0.54945 |
| model.layers.6 | 702,610,000 | 0.1005 | 0.5523 |
| model.layers.7 | 790,698,576 | 0.0887 | 0.56539 |
| model.layers.8 | 878,787,152 | 0.0859 | 0.5745 |
| model.layers.9 | 966,875,728 | 0.0947 | 0.58727 |

**Summary:** ~0.10 s per layer, **2.2 s total for the full 22-layer forward
pass**, checksums finite and monotonically increasing (real progressive
representation flow: 0.024 → 0.587).

## What this means

- The real model actually executes GEMM work per layer.  Synthetic calibration
  ("0.5 s of MACs per MB") is what a custom `work_factor` is tuned to; real GEMM
  with a real prompt is the honest term of comparison.
- At `seq=1` the per-layer GEMM is ~0.1 s because matrix shapes are tiny.  With a
  real prompt (batched tokens), the projection GEMMs grow proportionally, so real
  compute per layer is closer to the bandwidth-delay product and masking can rise
  to defensible levels — **the real model is worth benchmarking over the
  synthetic stand-in**.
- The masking ratio stays compute/bandwidth-dependent; do not generalise this
  seq=1 run to production cold-start TTFT.  Feed a real prompt length (multi
  tokens) for a production-like TTFT/masking number.

**Caveat:** single-token, CPU runtime.  CUDA was not active on this box (RTX 4050, 6
GB VRAM per profile); the engine selects CUDA automatically if a GPU exists.
