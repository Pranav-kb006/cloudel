<!--
STATUS: WRITTEN — architecture and SAFETENSORS random-access design are documented.
Measured results (2026-11-11) are reported in `docs/project_report.md` §8 and `results/benchmark.csv`.
Packages needed to read or build this document: NONE (Markdown only).
Owner: WS-1 `architect`. Files owned: docs/01_architecture.md, docs/02_math_model.md.
-->

# 01 — System Architecture

**Project:** Design and Empirical Evaluation of an Asynchronous Progressive Layer-Streaming Engine for Accelerating Serverless Large Language Model Invocations
**Course:** IS355TBD (Cloud Computing), 5th Semester, RVCE — Experiential Learning
**Document owner:** WS-1 (`architect`)
**Status:** Design fixed; implementation artifacts built by WS-2/WS-3 and exercised. Measured results are in `results/benchmark.csv` and `docs/project_report.md` §8.

> **STATUS:** This document specifies the architecture, the SAFETENSORS byte-range access model, the request lifecycle, failure handling, and the local-to-production substitution. It contains **no measured performance numbers**; the measured 2026-11-11 run is reported in `docs/project_report.md` §8 and `results/benchmark.csv`. Reading or building this document requires **no packages**.

---

## 1. Executive summary

Serverless LLM inference is dominated by cold-start weight loading: a worker must materialise gigabytes of parameters before it can emit a single token, so time-to-first-token (TTFT) is often an order of magnitude worse than steady-state decode. This project designs and empirically evaluates an **asynchronous progressive layer-streaming engine** that removes the "download everything, then compute" barrier. Model weights live in a single SAFETENSORS file in S3/MinIO. A small JSON manifest maps each transformer layer to an absolute byte range. Inside the serverless worker, a **PipelineDriver** executes layer $i$ while an **asyncio prefetch daemon** issues HTTP `Range` requests for layers $i+1$ and $i+2$ into a bounded **ring buffer**; a **memory-aware monitor** clamps buffer depth from live RSS; consumed layers are evicted immediately. The result is a three-stage asynchronous pipeline — client → worker-resident pipelined engine → object storage — in which network transfer and CPU compute overlap. The defensible claims are **lower TTFT** and **I/O masking**; peak-RSS reduction is **not** claimed because the naive implementation pre-allocates the full parameter set. On CPU, layer compute is the practical bottleneck for masking; the measured run achieved **62.6%** masking (layer-0 exposure only), below 70% and specific to the synthetic workload. All empirical details are in `results/benchmark.csv`.

---

## 2. Context: why serverless LLM cold start hurts TTFT

A serverless function is created on demand, runs one request (or a short burst), and is torn down. For an LLM this is hostile for three reasons:

1. **Weight volume.** Even a small open checkpoint is hundreds of megabytes to several gigabytes; a 7B model in FP16 is ~14 GB. The worker cannot compute until the parameters it needs are resident.
2. **Cold-start serialisation.** The conventional path is: schedule container → start runtime → download the whole checkpoint → deserialise/mmap → tokenise → run all layers → sample. The download and the forward pass are **serial**. If the download is $D$ seconds and the forward pass is $C$ seconds, TTFT is at least $D + C$ plus queueing and init.
3. **No reuse guarantee.** The next invocation may land on a different worker with an empty cache, so the download cost is paid repeatedly.

**Why layer-granular streaming helps TTFT.** Transformer inference is sequential across layers: layer $i$ cannot start before layer $i-1$ finishes, but the *bytes for layer $i+1$* can be fetched while layer $i$ is being computed. If layer downloads and layer computes are comparable, the download of layers $1 \ldots L-1$ can be hidden behind compute, leaving only the first layer's download exposed. The model is still fully consumed before the first token (an autoregressive decoder needs every layer for the prompt's last position), so streaming is an **overlap** technique, not a model-skipping technique. Its ceiling is therefore roughly a factor of two in the ideal case, not an order of magnitude — a point made precise in `docs/02_math_model.md`.

This work is positioned against serverless weight-loading systems (e.g. ServerlessLLM, LLM-in-a-Flash-style tiered loading, paged KV caches such as vLLM, and scheduling work such as FastServe/SOCK). The distinct contribution here is a **small, reproducible, end-to-end layer-streaming pipeline** with an explicit byte-range manifest, an async prefetcher, an RSS-aware ring buffer, and a benchmark harness that reports TTFT and masking honestly rather than assuming GPU-class overlap.

---

## 3. Design goals and non-goals

### 3.1 Goals

- G1 — Start computing layer 0 **before** the whole checkpoint is downloaded.
- G2 — Overlap layer $i$ compute with layer $i+1$ (and $i+2$) download using a bounded, backpressured buffer.
- G3 — Never exceed a configurable RSS budget; degrade lookahead rather than OOM.
- G4 — Address weights by **absolute byte range** so the object store can serve `206 Partial Content` without parsing the file.
- G5 — Produce reproducible TTFT, per-layer timing, masking-ratio, and RSS-trace artifacts from one command.
- G6 — Be honest: separate design claims from measured claims.

### 3.2 Non-goals (explicit)

- N1 — **No peak-RSS reduction claim.** The naive implementation pre-allocates the full parameter set; the ring buffer holds transient byte copies, not the only copy of the weights. See §10.
- N2 — No GPU kernels, no CUDA, no custom paged parameter store. CPU compute (with synthetic stand-in kernels in this build) is the execution model.
- N3 — No model training or fine-tuning. Only frozen/synthetic checkpoints are used.
- N4 — No multi-tenant scheduler, no request batching, no KV-cache paging. One request at a time in the worker.
- N5 — No claim that this is a production serverless deployment. The local RangeServer is a stand-in (see §9).

---

## 4. System overview

The system has three logical stages — **client → serverless worker → object store** — with the **API/gateway** acting as a control-plane hop between the first two. The client sends a prompt. The gateway admits the request and routes it to an ephemeral serverless worker. The worker hosts the **pipelined engine**, which owns the tokenizer/embeddings path, the `PipelineDriver`, the compute engine, the prefetch daemon, the ring buffer, and the RSS monitor. The engine pulls weight bytes on demand from **S3/MinIO** (locally: a Range-capable HTTP server over loopback).

```
┌──────────┐   HTTPS    ┌────────────────┐   invoke    ┌───────────────────────────────────────────────────────┐
│  Client  │───────────▶│ API / Gateway  │────────────▶│ Serverless worker (ephemeral; one request per instance)│
│ (prompt) │◀───────────│ auth, routing  │◀────────────│                                                       │
└──────────┘  tokens    └────────────────┘   stream     │  ┌─────────────────────────────────────────────────┐  │
                                                         │  │ Pipelined engine                                │  │
                                                         │  │                                                 │  │
                                                         │  │  prompt ──▶ Tokenizer ──▶ Embeddings            │  │
                                                         │  │                              │                  │  │
                                                         │  │                              ▼                  │  │
                                                         │  │                     ┌─────────────────┐         │  │
                                                         │  │                     │ PipelineDriver  │         │  │
                                                         │  │                     │  bind layer i   │         │  │
                                                         │  │                     │  run forward    │         │  │
                                                         │  │                     │  evict layer i  │         │  │
                                                         │  │                     └───┬─────────┬───┘         │  │
                                                         │  │             params      │         │ logits      │  │
                                                         │  │                         ▼         ▼             │  │
                                                         │  │                  ┌──────────┐  LM head + sample  │  │
                                                         │  │                  │ Compute  │  ──▶ first token    │  │
                                                         │  │                  │ engine   │                    │  │
                                                         │  │                  └────▲─────┘                    │  │
                                                         │  │       get(layer i)    │                          │  │
                                                         │  │                  ┌────┴──────────────┐           │  │
                                                         │  │                  │ Ring buffer       │           │  │
                                                         │  │                  │ (bounded, k slots)│           │  │
                                                         │  │                  └────▲──────────────┘           │  │
                                                         │  │        put(layer)     │   RSS samples            │  │
                                                         │  │                  ┌────┴──────────────┐           │  │
                                                         │  │                  │ Prefetch daemon   │◀── RSS    │  │
                                                         │  │                  │ lookahead i+1,i+2 │   monitor │  │
                                                         │  │                  └────┬──────────────┘           │  │
                                                         │  └───────────────────────┼──────────────────────────┘  │
                                                         └──────────────────────────┼─────────────────────────────┘
                                                                                    │ HTTP Range → 206 Partial
                                                         ┌──────────────────────────▼─────────────────────────────┐
                                                         │ S3 / MinIO object store                                │
                                                         │   model.safetensors   (weights, byte-addressable)      │
                                                         │   layer_manifest.json (layer → key, start, end)        │
                                                         └────────────────────────────────────────────────────────┘
```

**Stage 1 — Client.** Sends a prompt and consumes a token stream. It knows nothing about layers or ranges.

**Stage 1b — Gateway (control-plane hop).** Authenticates, applies admission control/queueing, and dispatches to a worker. The gateway is where queue time $t_q$ is incurred. In this build it is collapsed into the harness.

**Stage 2 — Serverless worker + pipelined engine.** The only stateful-for-the-request component. It fetches the manifest, prefetches weights, computes layers in order, and streams output. The worker is disposable; all durable state is in the object store.

**Stage 3 — Object store.** Holds the SAFETENSORS checkpoint and the precomputed manifest. It must support HTTP `Range` (S3 `GetObject` with a range header; MinIO supports the same). It is the only component that persists across invocations.

---

## 5. Component-by-component description

### 5.1 Ring buffer (`src/core/MemoryBuffer.py`)

The buffer is a **fixed-capacity circular pool of layer payload slots**, not a FIFO queue. Semantics:

- Capacity is $k$ slots; each slot holds at most one layer payload (bytes plus a parsed tensor view).
- A `put(layer_id, payload)` call takes a free slot, records `layer_id → slot`, and marks the slot *full*. If no slot is free, the producer **awaits** a free slot (backpressure) rather than growing the buffer.
- A `get(layer_id)` call awaits until that specific layer is present, returns the payload, and (after the consumer releases it) marks the slot *free* for reuse.
- Slots are reused modulo $k$; the `layer_id → slot` map, not the slot index, defines correctness. This is what allows **out-of-order completion**: if layer $i+2$ arrives before $i+1$, it occupies a different slot and the driver still blocks on `get(i+1)`.
- Capacity is measured in **both slots and bytes**: a payload larger than `slot_bytes` is rejected (or split at the manifest level), so the buffer cannot be silently over-filled.
- `stats()` exposes occupancy, high-water mark, waits, and evictions for the benchmark.

The buffer is intentionally *small* (lookahead + in-flight ≈ 3 slots by default). A large buffer would hide more jitter but would raise RSS and blur the memory-trajectory experiment.

### 5.2 Tokenizer and embeddings

The tokenizer converts the prompt into token IDs. Tokenisation itself needs **no weights** and can run concurrently with the first weight fetch. The embedding lookup, however, needs the token-embedding matrix (`model.embed_tokens.weight`), which is treated as a **pseudo-layer 0** (`embed`) in the manifest. The LM head (`lm_head.weight`) is treated as a **pseudo-layer L** (`head`). Thus the manifest covers *all* weight bytes, not just transformer blocks, and the pipeline cannot accidentally omit them. In this build the tokenizer is a deterministic stand-in (no external vocabulary download required); in production it would be the model's SentencePiece/BPE tokenizer.

### 5.3 Compute engine

The compute engine executes one layer's forward pass. It receives a bound parameter view (tensors decoded from the layer payload according to `dtype`/`shape`) plus the hidden state, and returns the new hidden state. In this build, **synthetic compute kernels** stand in for transformer GEMMs/attention so the artifact runs on the Python standard library without `torch`; this substitution is disclosed in the report and in `docs/02_math_model.md`. In production the same interface would wrap a real kernel (PyTorch/ONNX/llama.cpp). Compute time per layer is denoted $c_i$.

The engine is **single-threaded with respect to layers**: layer $i+1$'s compute cannot start before layer $i$'s compute finishes (autoregressive dependency). This is exactly why prefetch, not parallel compute, is the available overlap.

### 5.4 Prefetch daemon (`src/core/AsyncPrefetcher.py`)

An `asyncio` task (or small task pool) that:

- Reads the layer manifest and issues HTTP `GET` with `Range: bytes=start-(end-1)` for the **next** layer, with lookahead $N_{\text{look}}=2$ (i.e. layers $i+1$ and $i+2$ while computing $i$).
- Limits in-flight requests (`max_concurrency`) so it does not saturate the link or the buffer.
- Applies a **pace heuristic**: it may start layer $i+2$ only when the buffer has a free slot and the estimated remaining compute for $i$ exceeds the estimated download time of $i+1$; otherwise it waits, avoiding useless prefetch that would be evicted.
- Retries transient failures with exponential backoff and resumes from the last received byte using a new `Range` (see §7).
- Verifies length and, where present, a SHA-256 digest from the manifest before handing the payload to the buffer.
- Exposes `io_wait_s` (time the driver actually blocked on a layer) so masking can be computed without guessing.

Lookahead is **not** fixed at 2: the RSS clamp may reduce it to 1 or 0.

### 5.5 Manifest and byte-range index (`src/core/SafetensorsManifest.py`, `src/core/S3ManifestWriter.py`)

`SafetensorsManifest` parses only the SAFETENSORS **header** (see §6), producing a table of `name → {dtype, shape, data_offsets}` and converting each entry to an **absolute file range** `[start, end)`. It then groups tensors by top-level layer prefix (e.g. `model.layers.17.*` → layer `17`) plus the `embed` and `head` pseudo-layers.

`S3ManifestWriter` reads that table **once, offline**, and emits a compact JSON document:

```json
{
  "model": "synthetic-2layer",
  "file": "model.safetensors",
  "size": 67109169,
  "etag": "…",
  "layers": {
    "0": {"key": "model.safetensors", "start": 305, "end": 33554737,
          "length": 33554432, "tensors": ["model.layers.0.self_attn.q_proj.weight"]},
    "1": {"key": "model.safetensors", "start": 33554737, "end": 67109169,
          "length": 33554432, "tensors": ["model.layers.1.self_attn.q_proj.weight"]}
  }
}
```

At runtime the worker fetches this **small** manifest (kilobytes) instead of the multi-gigabyte data section. The manifest is versioned by object `etag`/`size`; a mismatch invalidates cached ranges. This is the single source of truth for "what bytes belong to layer $i$".

### 5.6 Memory-aware eviction and RSS monitor

A monitor samples resident set size (via `psutil` when available, otherwise `/proc/self/statm`) at a fixed interval and after each layer. The driver computes a safe buffer depth:

$$
B \le \min\left(k_{\max},\ \left\lfloor \frac{RSS_{\text{limit}} - RSS_{\text{base}} - A_{\max}}{s_{\text{layer}}} \right\rfloor \right)
$$

where $s_{\text{layer}}$ is the layer payload size and $A_{\max}$ is a reserve for activations and transient copies. If the clamp falls below the requested lookahead, the prefetcher reduces lookahead; if it falls below 1, the pipeline degrades to **synchronous single-layer mode** (fetch, compute, evict) rather than risk an OOM kill. Eviction happens immediately after a layer is consumed: the slot is freed and the payload reference dropped. **Caveat:** in the naive implementation the *parameters themselves* are pre-allocated separately, so eviction frees only the transient buffer copy; peak RSS is not reduced. See §10.

---

## 6. SAFETENSORS random access

### 6.1 File layout

A SAFETENSORS file is three concatenated regions:

```
┌──────────────────────┬───────────────────────────────┬─────────────────────────────────────┐
│ 8-byte header length │ JSON header (N bytes, UTF-8)  │ data section (tensor bytes)          │
│ little-endian u64 N  │ may be space-padded to 8-byte │ concatenated, offsets relative to    │
│                      │ alignment; keys = tensor names│ the start of this section            │
└──────────────────────┴───────────────────────────────┴─────────────────────────────────────┘
byte 0                 byte 8                          byte 8 + N
```

1. **Bytes 0–7:** an unsigned 64-bit integer $N$, **little-endian**, giving the JSON header length.
2. **Bytes 8 … 8+N−1:** a UTF-8 JSON object. It contains an optional `"__metadata__"` object (string → string, e.g. format/framework) and one entry per tensor. Each tensor entry has:
   - `dtype` — e.g. `"F16"`, `"BF16"`, `"F32"`, `"I8"`, `"U8"`, `"I32"`, `"I64"`, `"BOOL"`;
   - `shape` — array of dimensions;
   - `data_offsets` — `[begin, end)`, **relative to the start of the data section**.
3. **Bytes 8+N … EOF:** the data section. Tensor bytes are laid out contiguously; there may be gaps/alignment padding, which is why the parser must trust `data_offsets` rather than compute them from shapes alone.

`"__metadata__"` is reserved and is **not** a tensor. Parsers must skip it. The JSON may be padded with spaces to an 8-byte boundary; `json.loads` tolerates trailing whitespace.

### 6.2 Computing absolute byte ranges

Given header length $N$ and a tensor's relative offsets $[b, e)$:

$$
\text{abs\_start} = 8 + N + b, \qquad \text{abs\_end} = 8 + N + e
$$

The range is half-open, `[abs_start, abs_end)`, with length $e-b$. HTTP `Range` headers are **inclusive**, so the request must be:

```
Range: bytes=<abs_start>-<abs_end - 1>
```

The server should answer `206 Partial Content` with `Content-Range: bytes <abs_start>-<abs_end-1>/<file_size>`. A `200 OK` means the server ignored the range — the client must detect this and abort (or fall back to a full download, which defeats streaming).

### 6.3 Worked example (illustrative)

Suppose the header length is $N = 297 = \texttt{0x0129}$. The first eight bytes are:

```
offset 0x0000:  29 01 00 00 00 00 00 00     # little-endian u64: 0x0000000000000129 = 297
offset 0x0008:  7b 22 5f 5f 6d 65 74 61 ... # '{"__meta...'  start of 297-byte JSON header
offset 0x0131:  <first tensor byte>         # data section starts at 8 + 297 = 305 = 0x0131
```

The header JSON (abridged, one line in the file) might be:

```json
{
  "__metadata__": {"format": "pt"},
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "F16", "shape": [4096, 4096], "data_offsets": [0, 33554432]
  },
  "model.layers.1.self_attn.q_proj.weight": {
    "dtype": "F16", "shape": [4096, 4096], "data_offsets": [33554432, 67108864]
  }
}
```

For `F16`, one element is 2 bytes, so $4096 \times 4096 \times 2 = 33{,}554{,}432$ bytes = 32 MiB per tensor. Absolute ranges:

| Tensor | Relative `[b,e)` | Absolute `[8+N+b, 8+N+e)` | HTTP request |
|---|---|---|---|
| layer 0 `q_proj` | `[0, 33554432)` | `[305, 33554737)` | `Range: bytes=305-33554736` |
| layer 1 `q_proj` | `[33554432, 67108864)` | `[33554737, 67109169)` | `Range: bytes=33554737-67109168` |

The file size is $305 + 67{,}108{,}864 = 67{,}109{,}169$ bytes. Fetching layer 1 transfers exactly 32 MiB; the header is 305 bytes. **No part of the data section is read to discover offsets.**

### 6.4 Why this enables Range fetches without parsing the whole file

The header is $O(\#\text{tensors})$ — kilobytes to a few megabytes — while the data section is $O(\text{parameters})$ — gigabytes. The access pattern is therefore:

1. `Range: bytes=0-7` → 8 bytes → read $N$.
2. `Range: bytes=8-(8+N-1)` → header only → parse `dtype`/`shape`/`data_offsets`.
3. For each needed layer, `Range: bytes=abs_start-(abs_end-1)` → exactly that layer's bytes.

Because offsets are explicit and absolute-addressable, the client never needs to stream the file linearly, never needs to deserialise unrelated tensors, and can issue **concurrent, out-of-order, resumable** requests. This is precisely what S3 `GetObject` ranges and MinIO range requests provide, and it is the property that makes layer-granular prefetch possible. The precomputed manifest is an optimisation: it moves steps 1–2 offline so the runtime fetch is a single small JSON GET.

### 6.5 dtype widths used by the parser

| dtype | bytes/element | notes |
|---|---:|---|
| `BOOL`, `I8`, `U8` | 1 | |
| `I16` | 2 | |
| `F16`, `BF16` | 2 | BF16 is bit-pattern storage, not numerically identical to F16 |
| `I32`, `F32` | 4 | |
| `I64`, `F64` | 8 | |

The parser validates `e - b == prod(shape) * width(dtype)`; a mismatch is a corrupt manifest, not a recoverable read.

---

## 7. Request lifecycle

Let $L$ be the number of transformer layers (plus `embed`/`head` pseudo-layers), $d_i$ the download time and $c_i$ the compute time of layer $i$. The exact recurrence is derived in `docs/02_math_model.md`; the timeline is:

```
t=0        request arrives at gateway; queue + admission           (t_q)
           worker starts; runtime init; manifest GET (small)       (t_i, t_m)
t=t_s      manifest cached; tokenizer runs on prompt (no weights)
           prefetch(embed / layer 0) issued
t=t_s+d_0  layer 0 bytes arrive; driver binds params
           prefetch(layer 1) issued; prefetch(layer 2) queued
t=t_s+d_0+c_0
           layer 0 compute done; slot evicted; driver requests layer 1
           (layer 1 download has been running under c_0)
t=...      steady state: compute(i) overlaps download(i+1), download(i+2)
           ring buffer stays at k≈2–3 slots; RSS clamp may shrink lookahead
t=T_last   last layer compute done; LM head + sampling
t=T_first  FIRST TOKEN emitted  →  TTFT = T_first - t_request
           subsequent decode continues (not modelled here)
```

Key points:

- **The first layer is never overlapped.** Nothing can compute before layer 0 arrives, so $d_0$ is fully exposed. This is the unavoidable pipeline-fill cost.
- **Tokenisation is free overlap.** It needs no weights and runs while layer 0 downloads.
- **Steady state** adds $\max(d_i, c_i)$ per layer, not $d_i + c_i$.
- **Eviction is immediate** after consumption, freeing a slot for $i+2$ and keeping the buffer bounded.
- **TTFT ends at the first output token**, after the LM head and sampling — not after the last layer's bytes are downloaded.

All times above are symbolic. The measured 2026-11-11 run is in `results/benchmark.csv` and `docs/project_report.md` §8.

---

## 8. Failure modes and mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| **Short read** | Fewer bytes than `Content-Length` / manifest `length` | Compare received length to `end-start`; re-issue `Range` from `start + received`; fail after $n$ attempts. |
| **HTTP 416** (`Range Not Satisfiable`) | Requested range beyond EOF | Manifest stale or object replaced. Re-fetch bytes 0–7 and the header, rebuild ranges, compare `etag`/`size`; abort loudly if the object changed mid-run. |
| **Checksum mismatch** | SHA-256 in manifest ≠ payload digest | Discard payload, refetch once from the object store, then fail the request. Never compute on corrupt weights. |
| **Connection reset / timeout** | `aiohttp`/socket exception mid-body | Exponential backoff with jitter; resume from last byte with a new `Range`; cap attempts and surface `io_wait_s`. |
| **Buffer pressure** | Producer blocked, RSS near limit | RSS clamp reduces lookahead ($2 \to 1 \to 0$); if depth < 1, switch to synchronous fetch-compute-evict; never grow unbounded. |
| **Out-of-order layers** | Layer $i+2$ completes before $i+1$ | Buffer keyed by `layer_id` with a free list (not FIFO); `get(i)` awaits only the needed layer; no head-of-line blocking of completed layers. |
| **Duplicate delivery** | Same layer arrives twice | `put` is idempotent by `layer_id`; duplicate payloads are dropped and the slot returned. |
| **Server ignores Range (200)** | Full file starts streaming | Detect status `200` when `206` was requested; abort/fallback; the RangeServer must be tested for `206`. |
| **Manifest/object mismatch** | Offsets point at wrong bytes | Pin `etag` + `size` in the manifest; validate the first 8 bytes and header hash at startup. |
| **Slow first byte (TTFB)** | Large gap before `d_0` starts | Keep-alive connection, DNS/TLS pre-warm, issue layer 0 request before tokenisation finishes. |
| **Worker OOM kill** | Process dies under RSS limit | Conservative `A_max` reserve, lookahead clamp, `slot_bytes` cap, and a synchronous fallback mode. |

Failures are **expected** in a WAN object-store path, so the prefetcher treats them as retryable state-machine transitions rather than exceptions that kill the request.

---

## 9. Production mapping: what is real and what is substituted

This 24-hour build runs entirely on one machine. The substitution is deliberate and must be stated in the report:

| Concern | This build (24 h stand-in) | Production path |
|---|---|---|
| Object store | `src/storage/RangeServer.py` serving a local file over HTTP loopback | **S3** (or on-prem **MinIO**) with `GetObject` range requests |
| Byte ranges | HTTP `Range` / `206 Partial Content` | S3/MinIO range GET (same semantics) |
| Bandwidth throttling | Sleep/token-bucket throttle inside `RangeServer.py` | **`tc`** (`tbf`/`netem`) on the worker NIC, or cloud WAN shaping / egress limits |
| Serverless runtime | Local Python process, one invocation | AWS Lambda / Knative / OpenFaaS with cold-start semantics |
| RSS monitoring | `psutil` or `/proc/self/statm` | Container cgroup memory limit + platform metrics |
| Manifest storage | Local JSON file | S3 object with versioning/ETag; optionally cached at the edge |

**Explicit statement:** *local RangeServer + HTTP-over-loopback + in-process throttling is a stand-in for MinIO/S3 + `tc` bandwidth throttling.* The protocols and failure semantics are the same (`Range`, `206`, `416`, retries), so the pipeline logic transfers; the absolute latency numbers do not. Production would use MinIO (on-prem) or S3 (cloud) and `tc` to emulate or enforce WAN bandwidth. Results from loopback are therefore **mechanism evidence**, not a deployment benchmark.

---

## 10. Known limitations

1. **No peak-RSS reduction claim.** The naive implementation pre-allocates the full model parameter set (or mmaps the whole file), so the ring buffer holds *additional transient copies*, not the sole copy. Evicting a layer frees only the buffer slot. Peak RSS is therefore dominated by the pre-allocated parameters. A real RSS win requires a custom parameter store (paged/streamed parameters with `madvise(MADV_DONTNEED)`, or a GPU/CPU tiered store) and is listed as future work. The defensible wins are **TTFT** and **I/O masking**.
2. **CPU compute is the masking bottleneck.** With synthetic or CPU kernels, per-layer compute may be too small to cover a throttled download (so $d_i > c_i$ and masking is low) or so large that TTFT is compute-bound (so speedup saturates near 1). The measured run achieved **62.6%** masking (all but layer 0 hidden); **no claim above 70%** is made. The value is specific to the 8-layer synthetic workload and 8 Mbit/s link.
3. **Synthetic compute stand-in.** Kernels do not perform real transformer GEMMs; they emulate per-layer cost and data dependencies. Latency is therefore not representative of a real model on real hardware.
4. **Small/synthetic checkpoint.** A 2-layer synthetic model is used for the reproducible demo; conclusions do not automatically scale to 7B+ models.
5. **Single worker, single request, single stream.** No batching, no concurrent requests, no multi-tenant isolation, no KV-cache paging.
6. **Approximate bandwidth control.** Sleep-based throttling is coarser than `tc`; burstiness and TCP dynamics are not faithfully reproduced.
7. **First-layer exposure.** $d_0$ is always exposed; a cold manifest fetch adds another round trip unless cached.
8. **Manifest staleness.** Ranges are valid only for the exact object version; a replaced checkpoint requires a manifest rebuild.
9. **No authentication/TLS on the local RangeServer.** It is a test fixture, not a secure service.
10. **GIL/asyncio overhead.** Prefetch and compute share one Python process; payload copies and JSON parsing compete with compute, reducing achievable overlap.
11. **No quantisation or compression.** Weights are fetched at their stored dtype; 4-bit or compressed formats would change the tradeoff.
12. **No model training.** Frozen/synthetic checkpoints only.

---

## 11. How this maps to the code

| File | Responsibility |
|---|---|
| `src/core/SafetensorsManifest.py` | Parse the 8-byte little-endian header length and JSON header; expose `dtype`, `shape`, `data_offsets`; convert to absolute `[start, end)`; group tensors into layers and `embed`/`head`. |
| `src/core/S3ManifestWriter.py` | Offline writer: turn a parsed manifest into the compact `layer → {key, start, end, length, tensors}` JSON consumed at runtime. |
| `src/core/AsyncPrefetcher.py` | `asyncio` HTTP Range client; lookahead $i+1, i+2$; concurrency cap; pacing; retries/resume; length/checksum validation; backpressure into the buffer. |
| `src/core/MemoryBuffer.py` | Bounded ring/slot pool; `put`/`get`/release; `layer_id → slot` map; byte accounting; RSS-based depth clamp; `stats()`. |
| `src/core/PipelineDriver.py` | Bind layer parameters, run the forward pass in order, evict after consumption, await the specific next layer, collect `ttft_s`, per-layer times, `io_wait_s`, `compute_s`, `masking_ratio`, `rss_trace`. |
| `examples/minimal_stream.py` | Build a tiny 2-layer synthetic SAFETENSORS file, serve it with Range support, stream it layer-by-layer, print TTFT and per-layer RSS — the end-to-end proof. |
| `src/storage/RangeServer.py` | Standard-library HTTP server with `Range`/`206`/`416` support and a configurable bandwidth throttle (the MinIO/S3 + `tc` stand-in). |
| `src/bench/BenchmarkHarness.py` | Run monolithic baseline vs pipelined engine; measure TTFT, I/O masking, end-to-end latency, RSS trajectory; emit CSV and avg/percentiles. |
| `run.sh` | Orchestrate RangeServer + harness; clean up processes; produce the results CSV. |
| `setup.sh` | Install optional deps (`aiohttp`, `psutil`, `requests`) and generate the synthetic model. |

---

## 12. Glossary

- **TTFT** — time from request admission to the first emitted output token.
- **Layer streaming** — fetching weight bytes for layer $i$ on demand instead of the whole checkpoint up front.
- **I/O masking** — fraction of download time hidden behind compute.
- **Lookahead** — number of layers prefetched ahead of the one being computed.
- **Ring buffer** — fixed-capacity circular pool of payload slots with reuse and backpressure.
- **Pseudo-layer** — `embed` (token embedding matrix) or `head` (LM head), treated as addressable layers so the manifest covers all weights.
- **Absolute range** — `[8 + N + b, 8 + N + e)`, the byte span an object store must serve.
- **Cold start** — worker creation + runtime init + first weight access before any token can be produced.

---

*All performance quantities in this document are symbolic. The measured 2026-11-11 run is in `results/benchmark.csv`; the numbers in the report are measured, not estimated.*
