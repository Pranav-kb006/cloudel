# Project Submission — Asynchronous Progressive Layer-Streaming Engine

**Course:** IS355TBD (Cloud Computing), 5th Semester, RVCE — Experiential Learning
**Date:** 2026-11-11
**Artifact:** Python 3.14, stdlib-first; no GPU or deep-learning framework required
**Status:** Built, executed, and measured. Results are in `results/benchmark.csv`; no number in this document was invented.

> **One-line summary.** A safetensors checkpoint is treated as a set of per-layer byte
> ranges; an asyncio prefetcher fetches layer `i+1` over HTTP Range while the worker
> computes layer `i`. On a synthetic 8-layer model at 8 Mbit/s, full-forward TTFT fell
> from **15.70 s to 10.82 s** (1.45×) with **62.6%** I/O masking.

---

## 0. How to reproduce

```bash
# One package install command (setup.sh does this idempotently):
python3 -m pip install aiohttp psutil requests

bash setup.sh    # generates models/synthetic_2layer.safetensors + synthetic_8layer.safetensors
bash run.sh      # starts RangeServer, runs 3 baseline + 3 pipeline reps, writes results/
```

Outputs: `results/benchmark.csv`, `results/per_layer.csv`, `results/rss/`.
Minimal proof of the pipeline: `python3 examples/minimal_stream.py` (stdlib only).

---

## 1. Problem and contribution

A cold serverless LLM worker cannot emit a token until its weights arrive from object
storage. The conventional path is monolithic: provision worker → download the whole
checkpoint → allocate all parameters → run the forward pass. Download and compute are
serial, so

```
T_base = T_startup + D + C
```

and peak memory holds the full model.

**Question.** Can a worker begin computing early layers while later layers are still in
flight, reducing TTFT and overlapping I/O with compute, without a GPU-resident copy?

**Contributions actually built:**

1. A **stdlib-only safetensors random-access parser** — 8-byte little-endian header length,
   JSON tensor table, absolute `[start, end)` ranges, natural layer grouping.
2. A **layer manifest writer** mapping each layer to an object key and byte range.
3. An **asyncio prefetcher + pipelined driver** with lookahead `N+1`, adaptive pacing,
   bounded ring-style buffer, and RSS-aware clamp.
4. A **reproducible benchmark harness** reporting TTFT, masking, RSS trajectory, and
   per-layer overlap, with an explicit TBD path on failure.

The claim is deliberately narrow: **overlap and accounting**, not compression, not a new
scheduler, and not a memory-saving parameter store.

---

## 2. Architecture

Three-stage asynchronous pipeline:

```
Client ──invoke──▶ Serverless Worker (cold)                    Object Storage
       ◀─tokens──  Manifest / Layer Index                          safetensors
                   AsyncPrefetcher (asyncio, N+1) ──HTTP Range──▶  byte ranges
                   MemoryBuffer (bounded, RSS clamp)              (206 responses)
                   PipelineDriver: load → bind → forward → evict
```

- **Client** sends a prompt and consumes tokens; it knows nothing about layers.
- **Serverless worker** reads a small JSON manifest, prefetches ahead into a bounded
  buffer, computes layers in order, evicts each payload after use, and emits the first
  token only after the full forward pass.
- **Object storage** serves HTTP `Range` requests. In this evaluation it is a local
  stdlib RangeServer over loopback — an explicit stand-in for MinIO/S3.

Layer `i+1`'s transfer overlaps layer `i`'s compute; layer 0 is unavoidably exposed; the
buffer bounds *transient* payloads, not a pre-allocated parameter store. Every layer still
computes, so streaming is an **overlap technique**, not layer skipping.

**Safetensors random access.** Layout:
`[8-byte LE uint64 header length N][N bytes UTF-8 JSON][data section]`.
The JSON gives each tensor's `dtype`, `shape`, and `data_offsets` relative to the data
section. Absolute range = `8 + N + begin` to `8 + N + end`. HTTP `Range` can therefore
fetch one layer without downloading or parsing the whole checkpoint. The parser validates
duplicates, bounds, and 8-byte alignment; it rejects malformed headers.

---

## 3. Mathematical model

Symbols: `L` layers; `d_i` download time; `c_i` compute time; `D = Σd_i`; `C = Σc_i`;
`T_startup` provisioning/runtime/manifest overhead; `N_look` lookahead depth; `M` masking.

**Baseline (serial):**

```
T_base = T_startup + D + C
```

**Pipelined (exact recurrence):**

```
A_0 = d_0,            F_0 = d_0 + c_0
A_i = max(A_{i-1} + d_i, F_{i-1})
F_i = A_i + c_i
T_pipe = T_startup + F_{L-1}
```

**Closed form with residual stalls:**

```
T_pipe = T_startup + d_0 + Σ c_i + Σ_{i=1..L-1} max(0, d_i - c_{i-1})
```

**I/O masking ratio (layer 0 excluded because it cannot be overlapped):**

```
M = Σ_{i=1..L-1} min(d_i, c_{i-1}) / D
  = (D - d_0 - Σ residual stalls) / D
```

**Ceiling and speedup bound:**

```
M ≤ 1 - d_0/D          (this pipeline's structural ceiling)
S = T_base / T_pipe ≤ (D + C) / (d_0 + C) < 2
```

The `< 2×` bound is the key guardrail: the whole checkpoint still crosses the link and every
layer still computes. A measured speedup above ~2× would indicate a measurement error.

**Memory trajectory.** In the naive implementation the full parameter set `P` is
pre-allocated:

```
R_naive(t) = R_0 + P + Σ_{j∈W(t)} s_j
B_max ≤ N_look · max_i s_i
```

The ring buffer bounds only `W(t)`, the transient payload window. A true streaming design
would drop `P`, which requires a custom parameter store (future work). **No peak-RSS
saving is claimed.**

---

## 4. Implementation (actual files)

| Component | File | What is implemented |
|---|---|---|
| Safetensors parser | `src/core/SafetensorsManifest.py` | `struct` header length, JSON table, dtype/shape validation, absolute ranges, mmap header-only reads |
| Manifest writer | `src/core/S3ManifestWriter.py` | Layer → key + `[start,end)` + tensor metadata; atomic JSON write; optional boto3 upload |
| Async prefetcher | `src/core/AsyncPrefetcher.py` | asyncio producer/consumer, HTTP Range, `N+1` lookahead, adaptive pacing, retries/backoff, aiohttp with urllib fallback, 206 validation |
| Memory buffer | `src/core/MemoryBuffer.py` | Bounded FIFO slots, byte accounting, `max_depth` + `rss_limit_mb` clamp, pin/unpin, eviction logging |
| Pipeline driver | `src/core/PipelineDriver.py` | get → bind → deterministic synthetic FP32 forward → evict; TTFT, first-layer, I/O wait, compute, masking, RSS trace |
| Minimal proof | `examples/minimal_stream.py` | Builds a valid 2-layer safetensors file, serves it with Range, streams it, prints TTFT and per-layer RSS |
| Range server | `src/storage/RangeServer.py` | `ThreadingHTTPServer`, RFC 7233 `206`/`416`/HEAD, shared token-bucket throttle |
| Benchmark harness | `src/bench/BenchmarkHarness.py` | Monolithic vs pipelined in fresh subprocesses, RSS sampling, CSV + per-layer CSV + percentiles, explicit TBD on failure |
| Model generator | `src/bench/make_synthetic_model.py` | Valid synthetic safetensors + sidecar manifest |
| Scripts | `run.sh`, `setup.sh` | Stale-process cleanup, readiness poll, trap cleanup; idempotent dependency install and model generation |

**Stack:** Python 3.14, stdlib plus `aiohttp` (async Range), `psutil` (RSS), `requests`
(declared for parity). `torch`, `numpy`, and `safetensors` are **not** required. The compute
kernel performs real FP32 MAC/elementwise arithmetic over downloaded bytes, but it is
**not** a transformer.

---

## 5. Experimental methodology

- **Workload:** synthetic 8-layer F32 checkpoint, 8.00 MiB, 1 MiB per layer.
- **Link:** local RangeServer + HTTP loopback with an 8 Mbit/s shared throttle. This is the
  24-hour stand-in for **MinIO/S3 + `tc`/`netem`**; the protocol semantics transfer, the
  WAN/cold-start effects do not.
- **Baseline:** one full HTTP GET (200), then compute all layers from the downloaded body.
- **Pipelined:** manifest → layer `i` Range fetch → compute → evict, prefetching ahead with
  lookahead 2.
- **Runs:** 3 per mode, each in a fresh spawned process for RSS isolation; warm-up discarded
  by construction (each process is new).
- **TTFT definition:** completion of the **full forward pass**. A real autoregressive first
  token needs every layer, so this is the only defensible TTFT. `first_layer_s` is reported
  separately as streaming-fill latency and is not an LLM TTFT.
- **Metrics:** TTFT, first-layer latency, total, I/O wait, compute, masking, peak RSS, RSS
  trajectory, per-layer blocked time.

**Threats to validity.** Loopback ≠ production object storage; synthetic compute ≠
transformer GEMMs; single worker, single request, no batching/KV cache/GPU; no real
cold-start provisioning (`T_startup` is modelled, not measured); sleep-based throttle is
coarser than `tc`.

---

## 6. Measured results

Source: `results/benchmark.csv`, 2026-11-11. No values below are estimates.

| Run | Mode | TTFT (s) | First layer (s) | Total (s) | I/O wait (s) | Compute (s) | Masking M | Peak RSS (MB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 15.4816 | 9.6308 | 15.4816 | 8.6578 | 6.8199 | 0.0000 | 53.688 |
| 1 | pipeline | 11.0237 | 3.8590 | 11.0237 | 3.1357 | 7.8828 | 0.6262 | 47.902 |
| 2 | baseline | 15.6011 | 9.4411 | 15.6011 | 8.6527 | 6.9446 | 0.0000 | 53.738 |
| 2 | pipeline | 10.8122 | 4.0109 | 10.8122 | 3.1345 | 7.6744 | 0.6263 | 48.777 |
| 3 | baseline | 16.0240 | 9.6731 | 16.0240 | 8.6604 | 7.3588 | 0.0000 | 53.574 |
| 3 | pipeline | 10.6129 | 4.0073 | 10.6129 | 3.1387 | 7.4703 | 0.6258 | 48.172 |
| **Mean** | **baseline** | **15.7022** | **9.5817** | **15.7022** | **8.6570** | **7.0411** | **0.0000** | **53.667** |
| **Mean** | **pipeline** | **10.8163** | **3.9590** | **10.8163** | **3.1363** | **7.6758** | **0.6261** | **48.284** |
| p50 | baseline | 15.6011 | 9.6308 | 15.6011 | 8.6578 | — | 0.0000 | 53.688 |
| p50 | pipeline | 10.8122 | 4.0073 | 10.8122 | 3.1357 | — | 0.6263 | 48.172 |
| p95 | baseline | 15.9817 | 9.6689 | 15.9817 | — | — | 0.0000 | 53.730 |
| p95 | pipeline | 11.0026 | 4.0105 | 11.0026 | — | — | 0.6263 | 48.720 |

### Interpretation

- **TTFT:** 15.70 s → 10.82 s, **31.1% lower, 1.45× speedup**. Below the ~2× ceiling, as
  the model predicts, because the whole checkpoint still crosses the link.
- **Masking:** **62.6%**. The exposed time is almost exactly layer 0's 3.13 s download.
  Layers 1–7 have `blocked_s` ≈ 0.00002–0.00008 s, i.e. they arrive under compute. The
  measured value is close to the structural ceiling `1 - d_0/D ≈ 0.627`, and it is below
  70%. It is specific to this workload/link and must not be generalised to a real model.
- **First-layer fill:** 9.58 s → 3.96 s. This is the streaming-fill diagnostic, **not** TTFT.
- **Peak RSS:** 53.7 MB → 48.3 MB in this harness. This reflects the synthetic baseline
  holding the whole 8 MiB file while the pipeline holds a bounded window. It is **not** a
  transformer memory-saving claim; see limitations.

---

## 7. Viva essentials (full Q&A in `docs/viva_qa.md`)

**Q1 — Doesn't prefetch fully hide I/O?** No. Measured `M = 0.626`, not 100%, not 70%+.
Layer 0 must arrive before any compute, so the structural ceiling is `1 - d_0/D`; at
`d_0 ≈ 3.13 s`, `D ≈ 8.39 s`, that is 62.7%. We measured 62.6%.

**Q2 — Where is the memory saving if you pre-allocate all params?** There is none in the
naive implementation, and we do not claim one. The defensible wins are TTFT and I/O
masking. A true streaming memory win needs a custom parameter store (mmap arena or
per-layer allocate/free) or quantisation; that is future work. The measured 53.7 → 48.3 MB
gap is buffer depth / whole-file retention in the synthetic harness, not parameter-memory
streaming.

**Q3 — Why safetensors + HTTP Range?** The header gives exact byte ranges and is pure data
(no pickle execution). `mmap` works locally but cannot cross an object-store boundary;
Range requests let a remote worker fetch one layer with `206 Partial Content`.

**Q4 — How is this different from the five papers?** ServerlessLLM is GPU checkpoint
loading; LLM in a Flash streams from local flash with a predictor; vLLM/FastServe optimise
serving and scheduling after weights are resident; SOCK optimises container provisioning.
This work combines layer-granular Range streaming from remote object storage, async
overlap, and an RSS-clamped buffer at the cold-start boundary. Each mechanism is known; the
integration and measurement are the modest contribution.

**Q5 — Is this a real serverless deployment?** No. Local RangeServer + loopback is a
stand-in for MinIO/S3 + `tc`. It validates the protocol path and overlap accounting; it
does not measure Lambda/Cloud Run cold start, WAN jitter, cross-AZ cost, or multi-tenancy.
Production testing is the next step.

---

## 8. Publication viability

| Venue | Type | Fit |
|---|---|---|
| **IEEE CloudCom** | International, cloud computing | Strong: serverless cold start + object storage |
| **IEEE INDICON** | India-wide IEEE | Good: systems prototype with measured results |
| **IEEE DISCOVER** | IEEE computing/communication | Reasonable: broad scope, short paper |
| **IEEE CONFLUENCE** | IEEE emerging technologies | Reasonable: short paper / poster |

**Distinct contribution table**

| Work | What it does | What this project adds | Overlap / limitation |
|---|---|---|---|
| ServerlessLLM (OSDI'24) | GPU checkpoint loading, migration | Layer-granular Range streaming on a CPU worker | Shares cold-start motivation; weaker hardware story |
| LLM in a Flash (arXiv'23) | Flash weight streaming, neuron prediction | Remote object storage + explicit manifest | Shares overlap; no predictor |
| vLLM / PagedAttention (SOSP'23) | Paged KV cache, batching | Cold-start data path before serving | Complementary, not competing |
| FastServe (arXiv'23) | Request scheduling, preemption | Weight availability for the first request | Complementary, not competing |
| SOCK (ATC'18) | Container dependency prefetch | Model-internal weight streaming | Different granularity; shares cold-path framing |

**Path:** EL report/viva now → workshop/short paper after MinIO + `tc`, a frozen public
checkpoint (SmolLM/TinyLlama), and more repetitions. The current synthetic prototype alone
is not a full research paper; the honest novelty is the integration and the measured
accounting of CPU-worker overlap.

---

## 9. Known limitations (do not overclaim)

1. **Synthetic compute, not a transformer.** The kernel does real FP32 arithmetic over
   downloaded bytes, but no attention/GEMM/LLM accuracy is claimed. Absolute latencies are
   not transformer latency.
2. **Local RangeServer + loopback, not serverless.** No cold-start provisioning, WAN jitter,
   cross-AZ cost, TLS/auth, or multi-tenant contention. `T_startup` is modelled, not measured.
3. **No peak-RSS claim.** The naive implementation pre-allocates full parameters. The
   measured RSS gap is a harness artifact (whole-file baseline vs bounded buffer), not
   parameter-memory streaming.
4. **Masking is 62.6%, not 70%+.** It is close to `1 - d_0/D` for this workload and is
   specific to the 8-layer synthetic model and 8 Mbit/s link.
5. **One worker, one request, one HTTP request per layer.** No batching, KV-cache paging,
   GPU backend, sub-layer chunking, HTTP/2 multiplexing, or compression/quantisation.
6. **Throttle is sleep-based**, coarser than `tc`; burstiness and TCP dynamics are not
   faithfully reproduced.
7. **S3 writer upload path is untested against MinIO/S3**; local manifest writing and
   Range serving are tested.
8. **No full model was trained or evaluated.** Frozen SmolLM/TinyLlama is future work and
   no latency number for those models is claimed.

---

## 10. Verification evidence

- `python3 -m py_compile src/core/*.py src/storage/*.py src/bench/*.py examples/*.py` — OK.
- `python3 examples/minimal_stream.py` — exit 0; 2-layer synthetic model, Range server,
  full-forward TTFT 76.88 ms, first-layer 36.24 ms, per-layer RSS printed.
- `Range: bytes=100-199` → 206, exact 100 bytes; `bytes=999999999-` → 416 with
  `Content-Range: bytes */8388936`; HEAD and suffix ranges verified.
- Throttle: 1 MiB at 8 Mbit/s ≈ 1.084 s; two concurrent 1 MiB ≈ 2.120 s (shared limiter).
- `bash setup.sh` twice — idempotent; `bash run.sh` exit 0, no stale server left, CSV written.
- Failure path: dead URL → exit 2 and a CSV row marked
  `TBD — pending successful run: ConnectionRefusedError...`.
- `bash scripts/verify_build.sh` — structural/compile/minimal-example checks.

**Final measured summary (3 runs/mode):** TTFT **15.702 s → 10.816 s** (1.45×, 31.1%
lower), masking **0.626**, first-layer fill **9.582 s → 3.959 s**, peak RSS **53.7 MB →
48.3 MB** (reported, not claimed as a memory win).
