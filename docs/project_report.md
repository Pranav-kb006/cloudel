STATUS: Report COMPLETE with measured results. The benchmark ran on 2026-11-11 against the real core APIs (`results/benchmark.csv`, 3 runs/mode, synthetic 8-layer model, 8 Mbit/s shared throttle). No result was fabricated. Code-path references were verified against the final `src/` tree. No Python packages are needed to read these docs.

---

# Design and Empirical Evaluation of an Asynchronous Progressive Layer-Streaming Engine for Accelerating Serverless Large Language Model Invocations

**Course:** IS355TBD — Cloud Computing
**Semester:** 5th Semester
**Institution:** Rashtreeya Vidyalaya College of Engineering (RVCE), Bengaluru
**Project type:** Experiential Learning (EL) Project
**Date:** 2026-11-11
**Team:** ____

---

## Abstract

Serverless platforms are attractive for large language model (LLM) inference because they promise pay-per-invocation elasticity and zero idle cost, but they expose a harsh cold-start path: a freshly provisioned worker must fetch tens to hundreds of gigabytes of model weights from remote object storage before it can emit its first token. The dominant practice — download the entire checkpoint, then run the forward pass — serialises network transfer and computation, inflating time-to-first-token (TTFT) and holding peak memory for the whole model. This project designs, implements, and empirically evaluates an asynchronous progressive layer-streaming engine that treats a safetensors checkpoint as a set of random-access byte ranges, fetches layers over HTTP Range requests, and overlaps layer $i{+}1$'s transfer with layer $i$'s compute through an asyncio prefetcher, a bounded ring buffer, and an RSS-aware clamp. We describe a three-stage architecture, a latency model that decomposes TTFT into startup, download, compute, and residual stall terms, and a benchmark methodology comparing monolithic and pipelined execution on a local Range server with configurable bandwidth. Because the evaluation environment has no GPU and no deep-learning framework, layer compute is a deterministic synthetic stand-in for transformer GEMMs; this substitution is disclosed and bounded. We report TTFT, end-to-end latency, I/O masking ratio, and peak RSS, and we state explicitly which claims the current implementation can and cannot support.

**Keywords:** serverless computing, cold start, large language models, safetensors, HTTP Range, asynchronous prefetching, I/O masking, time-to-first-token.

---

## 1. Introduction and Problem Statement

### 1.1 Motivation

Serverless computing decouples the unit of deployment from the unit of provisioning: an application is uploaded as a function or container image, and the provider schedules it onto a worker on demand. For bursty, latency-sensitive inference this model is economically attractive — a tenant pays only for the milliseconds in which a worker is actually executing. The same property, however, makes the *first* invocation of a function fundamentally different from the steady-state invocation. The provider must locate a worker, pull the container image, initialise the runtime, and only then begin application work. This is the *cold-start* problem, and it is well documented to dominate tail latency for short functions [5], [6].

LLM inference makes the cold start materially worse than for a conventional function. A small language model may occupy several gigabytes of weights; a mid-sized one tens of gigabytes. In a serverless setting the weights cannot be assumed to be resident on the worker: they live in object storage (S3, MinIO, GCS) and must be transferred over the network. The conventional pipeline is monolithic:

1. provision the worker and initialise the runtime (startup cost $T_{\text{startup}}$);
2. download the complete checkpoint (total download $D$);
3. allocate the full parameter set;
4. execute the forward pass (total compute $C$);
5. emit the first token.

The time-to-first-token (TTFT) of this pipeline is approximately $S + D + C$. Because steps 2 and 3 are strictly ordered before step 4, the network and the CPU/GPU never work at the same time. For a model whose download is comparable to its compute — a realistic regime on commodity hardware and on throttled serverless egress — roughly half of the cold-start budget is spent waiting for bytes that the compute unit could have been processing.

### 1.2 Problem statement

> Given a safetensors LLM checkpoint stored in an object store, can a serverless worker begin computing on early layers while later layers are still in flight, thereby reducing TTFT and overlapping I/O with compute, without requiring a GPU-resident copy of the entire model?

The engineering question has three parts:

- **Access granularity.** Can the checkpoint be addressed at layer granularity without parsing the whole file, and can those ranges be fetched individually from object storage?
- **Concurrency.** Can transfers be issued ahead of demand and overlapped with compute, and how much of the transfer is actually hidden?
- **Memory.** Can the resident working set be bounded, and if not, what is the honest cost of that limitation?

### 1.3 Contributions

This project makes four concrete contributions:

1. A **safetensors random-access layer** that parses the 8-byte little-endian header length, the JSON tensor table, and the data-section offsets, exposing exact byte ranges per tensor and per layer.
2. A **manifest format** that maps each layer to an object key and byte range, so a worker can issue HTTP Range requests without downloading the file.
3. An **asynchronous prefetcher and pipelined driver** with a ring buffer, lookahead depth $N{+}1$, a pacing heuristic, and an RSS-aware clamp.
4. A **reproducible benchmark harness** comparing monolithic and pipelined execution and reporting TTFT, end-to-end latency, I/O masking ratio, and RSS trajectory.

We are equally explicit about what is *not* claimed: the current naive implementation still pre-allocates the full parameter set, so it does **not** reduce peak RSS; the compute kernel is synthetic; and the storage backend is a local Range server rather than production S3. Section 9 and Section 10 treat these limitations as first-class results rather than footnotes.

---

## 2. Literature Review

Five works frame the design space: one addresses serverless LLM checkpoint loading (ServerlessLLM), one addresses memory-constrained inference by streaming weights from flash (LLM in a Flash), two address serving-side memory and scheduling (vLLM/PagedAttention and FastServe), and one addresses the container-level cold-start path (SOCK).

### 2.1 ServerlessLLM (Fu et al., OSDI 2024) [1]

ServerlessLLM observes that serverless LLM inference is bottlenecked by loading model checkpoints from remote storage into GPU memory, not by the forward pass itself. It proposes a checkpoint-loading system that exploits the parallelism of local and remote storage, plus live migration of inference state to reduce cold-start latency. Its central insight — that checkpoint *loading* deserves systems-level optimisation distinct from model execution — directly motivates this project. The difference is locus and granularity: ServerlessLLM is GPU-centric and optimises how a complete checkpoint is staged into device memory, whereas we ask whether a CPU worker can begin executing *before* the checkpoint is complete by streaming individual layers.

### 2.2 LLM in a Flash (Alizadeh et al., 2023) [2]

This Apple work targets inference when the model does not fit in device DRAM. It streams parameter data from flash storage, predicts which neurons will be needed, and overlaps I/O with computation so that only a fraction of the weights is resident at once. It is the closest conceptual relative of our work: both treat weights as streamable data and both rely on prefetch to hide transfer latency. The differences are the storage tier (local flash vs. remote object storage), the granularity (neuron/row-level with a predictor vs. layer-level with a static manifest), and the deployment model (a single memory-constrained device vs. an elastic serverless worker). LLM in a Flash assumes low-latency local flash; a serverless worker cannot.

### 2.3 vLLM / PagedAttention (Kwon et al., SOSP 2023) [3]

vLLM identifies that serving throughput is limited by KV-cache memory fragmentation and proposes PagedAttention, which stores the KV cache in non-contiguous blocks managed like virtual-memory pages, plus continuous batching. It is a landmark in *serving-side* memory efficiency. It does not, however, address the cold-start weight-loading path: vLLM assumes the model is already resident on the serving node. Our work is complementary — vLLM would be the steady-state engine behind our cold-start loader, not a substitute for it.

### 2.4 FastServe (Wu et al., 2023) [4]

FastServe targets distributed LLM inference serving and proposes a skip-join multi-level feedback queue scheduler that preempts and migrates requests to minimise queueing delay, with a GPU memory management scheme for KV cache. Like vLLM, it is scheduler-centric: it optimises *which request runs next* once weights are loaded. It does not reduce the cost of getting weights onto a new worker. Our contribution operates one layer below FastServe's abstraction — on the data path that must complete before any scheduler decision matters.

### 2.5 SOCK (Oakes et al., USENIX ATC 2018) [5]

SOCK studies serverless cold starts at the container level, showing that provisioning and package-loading dominate function latency and that caching and prefetching container dependencies can substantially reduce it. It establishes the vocabulary and the measurement discipline we adopt (cold path vs. warm path, provisioning overhead $T_{\text{startup}}$). Its granularity is the container image and its dependency closure; ours is an individual weight layer inside a checkpoint. SOCK would reduce the time to get our engine *running*; our engine reduces the time for that running engine to produce a token.

### 2.6 Comparison

| Work | Core mechanism | Target bottleneck | Granularity | What it does **not** solve |
|---|---|---|---|---|
| **ServerlessLLM** [1] | Parallel checkpoint loading, live migration | Checkpoint load into GPU memory | Whole checkpoint / model | Layer-level streaming on CPU; overlap of transfer with per-layer compute |
| **LLM in a Flash** [2] | Flash-resident weights, neuron prediction, I/O–compute overlap | Device DRAM capacity | Neuron / row | Remote object storage; serverless provisioning |
| **vLLM / PagedAttention** [3] | Paged KV cache, continuous batching | KV-cache fragmentation, throughput | KV blocks / sequences | Cold-start weight loading |
| **FastServe** [4] | Skip-join MLFQ scheduling, KV preemption | Queueing delay under load | Request / token | Weight transfer before first request |
| **SOCK** [5] | Container dependency caching and prefetch | Container provisioning latency | Container image / package | Model-internal weight streaming |
| **This work** | Layer-granular Range streaming + async prefetch + RSS clamp | Cold-start TTFT on serverless workers | Individual layer byte range | True peak-RSS reduction (naive impl); production network effects |

Two additional references inform the evaluation. Hellerstein et al. [6] articulate the structural limits of serverless computing — including the cost of state transfer — and motivate measuring the cold path separately from the warm path. FlashAttention [7] is an I/O-awareness argument at the kernel level: it shows that minimising data movement, not just arithmetic, is a first-order optimisation. Our work applies the same I/O-awareness principle one level up, at the checkpoint layer.

---

## 3. Literature Gap

The five works above partition the problem cleanly, and the partition itself reveals the gap:

- **SOCK [5]** optimises the *container* cold start — getting the runtime and its dependencies onto a worker. It stops at the process boundary and says nothing about what the process does with a multi-gigabyte checkpoint.
- **ServerlessLLM [1]** optimises *checkpoint loading*, but for GPU-resident execution and at whole-checkpoint granularity. It does not expose per-layer byte ranges to a CPU worker or overlap transfer with layer-by-layer compute.
- **LLM in a Flash [2]** overlaps I/O and compute, but against *local flash* and with a learned neuron predictor, not against *remote object storage* addressed by an explicit manifest.
- **vLLM [3]** and **FastServe [4]** optimise *serving* — KV-cache layout and request scheduling — and explicitly assume the weights are already loaded.

No work in this set combines all four of the following in one design:

1. **layer-granular random access** to a safetensors checkpoint via parsed byte ranges;
2. **remote object-storage retrieval** using HTTP Range requests, without downloading the file;
3. **asynchronous prefetch / compute overlap** with a bounded lookahead;
4. **RSS-aware buffering** appropriate to a memory-constrained serverless worker.

That intersection — *layer-granular streaming from object storage with async overlap and memory-aware buffering, aimed at serverless cold-start TTFT* — is the gap this project addresses. It is a narrow, systems-level gap, and we do not claim it is unoccupied in adjacent forms; it is simply not the design point of any of the five works.

---

## 4. Proposed Architecture

### 4.1 Overview

The engine is a three-stage pipeline. A **client** issues an invocation. A **serverless worker** hosts the pipelined engine: a manifest reader, an asynchronous prefetcher, a bounded ring buffer, and a compute driver. **Object storage** (S3/MinIO in production, a local Range server in this evaluation) holds the safetensors checkpoint and serves byte ranges.

```
                 invocation
   +--------+   ----------->   +-------------------------------------------+        +------------------+
   | Client |                  |        Serverless Worker (cold)           |        |  Object Storage  |
   +--------+   <-----------   |                                           |        |  (S3 / MinIO)    |
        ^         first token  |  +-------------+    +------------------+  |  HTTP  |                  |
        |                      |  | Manifest /  |    |  AsyncPrefetcher |  | Range  |  checkpoint.     |
        |                      |  | Layer Index |--->|  (asyncio, N+1)  |<-+--------|  safetensors     |
        |                      |  +-------------+    +--------+---------+  |        |  (byte ranges)   |
        |                      |                              | put        |        +------------------+
        |                      |                     +--------v---------+  |
        |                      |                     |   Ring Buffer    |  |
        |                      |                     |  (bounded depth) |  |
        |                      |                     +--------+---------+  |
        |                      |                              | get        |
        |                      |                     +--------v---------+  |
        |                      |                     | PipelineDriver   |  |
        |                      |                     | load -> bind ->  |  |
        |                      |                     | forward -> evict |  |
        |                      |                     +------------------+  |
        |                      +-------------------------------------------+
        |                                        |
        +----------------------------------------+   TTFT, e2e, RSS trace
```

**Stage 1 — Manifest resolution.** `src/core/SafetensorsManifest.py` reads the first 8 bytes of the checkpoint as a little-endian unsigned integer, the header length, then parses that many bytes of JSON. The JSON maps tensor names to `dtype`, `shape`, and `data_offsets` relative to the start of the data section. The parser converts these into absolute file byte offsets and groups tensors by top-level layer prefix.

**Stage 2 — Manifest emission.** `src/core/S3ManifestWriter.py` consumes the parsed manifest and emits a JSON document mapping each layer name to an object key and a `[start, end)` byte range, together with the tensor list and total length. This decouples the worker from safetensors internals: at cold start it needs only the manifest, not the checkpoint.

**Stage 3 — Streaming and execution.** `src/core/AsyncPrefetcher.py` issues HTTP Range requests for layers ahead of demand into a ring buffer of configurable depth, using lookahead $N{+}1$ and a pace heuristic that avoids over-running the buffer. `src/core/MemoryBuffer.py` enforces a maximum buffer depth and an RSS ceiling, clamping prefetch when resident memory approaches the limit. `src/core/PipelineDriver.py` pulls layer $i$, binds its parameters, runs a forward step, and evicts the layer's payload before moving on. `src/storage/RangeServer.py` serves byte ranges with a configurable bandwidth throttle, and `src/bench/BenchmarkHarness.py` runs the monolithic and pipelined modes and writes a CSV.

### 4.2 Safetensors random access

A safetensors file is `[8-byte LE header length][JSON header][data section]`. The JSON header is self-describing and, critically, contains explicit `data_offsets` for every tensor. This makes random access a parsing problem rather than a search problem: no tensor is located by scanning, and no code execution is involved (unlike pickle-based checkpoints). Given a layer's tensor list, the byte range to fetch is the union of its tensors' absolute ranges — typically a contiguous span because tensors are written layer by layer.

### 4.3 Why a ring buffer

A ring buffer provides O(1) insertion and eviction and a hard bound on resident layers. Combined with the RSS clamp, it converts an unbounded prefetch stream into a bounded working set. The buffer depth is the primary tuning knob trading memory for overlap: depth $N_{\text{look}}$ holds at most $N_{\text{look}}$ layer payloads, so peak buffered bytes scale as $N_{\text{look}} \cdot \bar{s}$ where $\bar{s}$ is the mean layer size.

### 4.4 Component-to-file map

| Component | File | Responsibility |
|---|---|---|
| Safetensors parser | `src/core/SafetensorsManifest.py` | Header length, JSON table, absolute byte ranges, layer grouping |
| Manifest writer | `src/core/S3ManifestWriter.py` | Layer → object key + `[start, end)` + tensor list |
| Async prefetcher | `src/core/AsyncPrefetcher.py` | HTTP Range, asyncio tasks, lookahead $N{+}1$, pacing |
| Memory buffer | `src/core/MemoryBuffer.py` | Bounded ring buffer, RSS clamp |
| Pipeline driver | `src/core/PipelineDriver.py` | Load → bind → synthetic forward → evict; emits metrics |
| Range server | `src/storage/RangeServer.py` | Stdlib HTTP server, Range support, bandwidth throttle |
| Benchmark harness | `src/bench/BenchmarkHarness.py` | Monolithic vs. pipelined, CSV, percentiles |
| Minimal demo | `examples/minimal_stream.py` | 2-layer synthetic model, local server, TTFT/RSS print |

---

## 5. Mathematical Model

We use the following symbols consistently with `docs/02_math_model.md`. For a checkpoint with $L$ layers:

- $T_{\text{startup}}$ — startup/provisioning overhead (worker acquisition, runtime init, connection setup);
- $d_i$ — time to download layer $i$;
- $c_i$ — time to compute layer $i$;
- $D = \sum_{i=1}^{L} d_i$ — total download time;
- $C = \sum_{i=1}^{L} c_i$ — total compute time;
- $M$ — I/O masking ratio, the fraction of download time hidden behind compute.

### 5.1 Baseline TTFT

The monolithic path downloads everything, then computes everything, and the first token requires a full forward pass:

$$
T_{\text{base}} = T_{\text{startup}} + D + C.
$$

### 5.2 Pipelined TTFT

With progressive streaming, layer 1 must arrive before compute starts; thereafter layer $i$'s transfer can overlap layer $i{-}1$'s compute. The residual stall on layer $i$ is $\max(0,\, d_i - c_{i-1})$, so

$$
T_{\text{pipe}} = T_{\text{startup}} + d_1 + \sum_{i=1}^{L} c_i + \sum_{i=2}^{L} \max\!\big(0,\, d_i - c_{i-1}\big).
$$

The ideal case is $d_i \le c_{i-1}$ for all $i \ge 2$, giving the lower bound

$$
T_{\text{pipe}}^{\star} = T_{\text{startup}} + d_1 + C.
$$

With finite lookahead $N_{\text{look}}$, a layer's fetch may start later than the ideal schedule; we model this as an additive penalty $\Delta_{N_{\text{look}}} \ge 0$:

$$
T_{\text{pipe}}^{(N_{\text{look}})} = T_{\text{startup}} + d_1 + C + \sum_{i=2}^{L}\max\!\big(0,\, d_i - c_{i-1}\big) + \Delta_{N_{\text{look}}},
\qquad \Delta_{N_{\text{look}}} \to 0 \text{ as } N_{\text{look}} \to L.
$$

### 5.3 I/O masking ratio

$$
M = \frac{T_{\text{base}} - T_{\text{pipe}}}{D}
  = \frac{D - d_1 - \sum_{i=2}^{L}\max\!\big(0,\, d_i - c_{i-1}\big)}{D},
\qquad 0 \le M \le 1.
$$

$M = 1$ means every byte after the first layer arrived while the compute unit was busy; $M = 0$ means no overlap at all.

### 5.4 Amdahl-like limit and the honest correction

At most $C$ seconds of transfer can be hidden behind $C$ seconds of compute, so

$$
M \le M_{\text{ideal}} = \min\!\left(1,\, \frac{C}{D}\right).
$$

This bound is necessary but not sufficient. It assumes the prefetcher can make progress *concurrently* with compute. In a single-process Python worker, CPU-bound layer compute can contend with the asyncio event loop, so transfers may issue late and overlap may degrade. We therefore introduce an overlap efficiency $\eta \in (0, 1]$:

$$
M_{\text{eff}} = \eta \cdot M_{\text{ideal}}, \qquad \eta = \eta(\text{compute contention}, \text{request overhead}, \text{buffer stalls}).
$$

$\eta$ is an empirical property of the implementation, not a constant we may assume. In the measured run, $M=0.626$ (62.6%), close to the geometric ceiling $1-d_0/D$: layer 0's 3.13 s download is exposed while all later layers arrive under compute. Compute contention did not dominate here because the synthetic kernel runs in a worker thread and socket I/O releases the GIL; the exposed time is structural, not an overlap failure. This is below 70% and should be read as mechanism evidence on a synthetic workload, not a general CPU claim. The measured value comes from `results/benchmark.csv`; nothing here is estimated.

### 5.5 Memory trajectory

Let $R_0$ be baseline process RSS, $P$ the total parameter bytes, $s_i$ the size of layer $i$, and $W(t)$ the set of layers resident at time $t$. The naive implementation pre-allocates the full parameter set, so

$$
R_{\text{naive}}(t) = R_0 + P + \sum_{j \in W(t)} s_j,
\qquad P = \sum_{i=1}^{L} s_i.
$$

A true streaming implementation would drop the $P$ term:

$$
R_{\text{stream}}(t) = R_0 + \sum_{j \in W(t)} s_j,
\qquad |W(t)| \le N_{\text{look}}.
$$

The gap between these two expressions is exactly the memory claim we do **not** make: the current artifact is $R_{\text{naive}}$, so peak RSS is not reduced. Peak buffered bytes are bounded by

$$
B_{\max} \le N_{\text{look}} \cdot \max_i s_i,
$$

which is the quantity the ring buffer and RSS clamp actually control.

### 5.6 Chunk-granularity tradeoff

For a chunk of $s$ bytes at bandwidth $BW$ with per-request overhead $\lambda$,

$$
d(s) = \frac{s}{BW} + \lambda.
$$

Very small chunks pay $\lambda$ repeatedly; very large chunks leave nothing to overlap. To fit a transfer inside the compute window of the preceding layer, choose

$$
s^{\star} = \min\!\left(s_i,\; c_i \cdot BW\right),
$$

i.e. a chunk no larger than what compute can cover. Layer granularity is the natural operating point because it matches the unit of execution; finer chunking would require intra-layer pipelining that the current driver does not implement.

---

## 6. Implementation Plan

### 6.1 Stack

- **Language:** Python 3.14.
- **Standard library:** `struct`, `json`, `asyncio`, `http.server`, `urllib`, `mmap`, `array`, `time`, `statistics`.
- **Third-party:** `aiohttp` (async HTTP Range client), `psutil` (RSS sampling), `requests` (harness convenience). These are installed by `setup.sh`.
- **Explicitly not required:** `torch`, `numpy`, `safetensors`. The parser is hand-written against the format; the compute kernel is a deterministic synthetic stand-in for transformer GEMMs. No model is trained.

### 6.2 Modules and status

| Path | Purpose | Status |
|---|---|---|
| `src/core/SafetensorsManifest.py` | Parse header length + JSON; expose per-tensor and per-layer absolute byte ranges | Implemented + tested |
| `src/core/S3ManifestWriter.py` | Write layer → object key + `[start, end)` manifest | Implemented + tested (local JSON; S3 upload untested) |
| `src/core/AsyncPrefetcher.py` | Asyncio prefetcher, HTTP Range, lookahead $N{+}1$, pace heuristic | Implemented + tested (aiohttp + urllib fallback) |
| `src/core/MemoryBuffer.py` | Bounded ring buffer + RSS clamp | Implemented + tested |
| `src/core/PipelineDriver.py` | Load, bind, synthetic forward, evict; emit metrics | Implemented + tested |
| `examples/minimal_stream.py` | 2-layer synthetic safetensors + local Range server + TTFT/RSS demo | Implemented + runs (exit 0) |
| `src/storage/RangeServer.py` | Stdlib HTTP Range server with bandwidth throttle | Implemented + tested (206/416/HEAD/suffix) |
| `src/bench/BenchmarkHarness.py` | Monolithic vs. pipelined benchmark, CSV + percentiles | Implemented + produced `results/benchmark.csv` |
| `run.sh` | Kill stale servers, run the harness, emit `results/benchmark.csv` | Implemented + exit 0 |
| `setup.sh` | Install `aiohttp`/`psutil`/`requests`, generate synthetic models | Implemented + idempotent |
| `requirements.txt` | Pinned third-party dependencies | Present |
| `docs/01_architecture.md` | Architecture detail (WS-1) | Present (403 lines) |
| `docs/02_math_model.md` | Canonical math model (WS-1) | Present (581 lines) |
| `docs/project_report.md` | This report | This file |
| `docs/viva_qa.md` | Viva preparation | Companion file |

> **Verification note.** All paths above exist and were verified on 2026-11-11. `bash scripts/verify_build.sh` reports ALL CHECKS PASSED; `python3 -m py_compile` passes; `examples/minimal_stream.py` exits 0; `bash run.sh` produced `results/benchmark.csv`. The S3 upload path is implemented but not tested against a live MinIO/S3 endpoint.

### 6.3 Setup and run

```bash
# one-time environment setup (idempotent)
bash setup.sh

# generate the synthetic checkpoint, start the Range server, run both modes
bash run.sh

# minimal end-to-end proof of the pipeline (stdlib only)
python3 examples/minimal_stream.py
```

`setup.sh` installs `aiohttp`, `psutil`, and `requests` and generates both synthetic safetensors models (2-layer minimal artifact and 8-layer benchmark model). It does **not** install `torch` or `numpy`, because neither is used. `run.sh` terminates any stale `RangeServer` process, starts a fresh one, executes the harness in monolithic and pipelined modes against the 8-layer model, and writes `results/benchmark.csv` with per-run metrics.

### 6.4 Configuration knobs

| Knob | Meaning | Default (per harness) |
|---|---|---|
| `--lookahead N` | Prefetch depth ahead of demand | 2 (effective window `N+1`) |
| `--rss-limit-mb` | RSS ceiling for the clamp | 1024 MB |
| `--bandwidth-mbps` | Range-server throttle | 8.0 Mbit/s |
| `--runs` | Repetitions per mode | 3 |
| `--work-factor` | Synthetic compute intensity | 1.0 |
| `--mode` | `both`, `baseline`, or `pipeline` | `both` |

The values above are the actual CLI defaults in `src/bench/BenchmarkHarness.py`; `run.sh` passes the same values.

---

## 7. Experimental Methodology

### 7.1 Design

We compare two modes on identical hardware, checkpoint, and bandwidth:

- **Baseline (monolithic):** download the complete checkpoint, allocate all parameters, then run the forward pass.
- **Pipelined (progressive):** stream layer by layer with lookahead $N{+}1$, overlapping layer $i{+}1$'s transfer with layer $i$'s compute.

Both modes use the same synthetic checkpoint and the same synthetic compute kernel, so any difference is attributable to the streaming schedule rather than to arithmetic.

### 7.2 Testbed substitution

The evaluation uses `src/storage/RangeServer.py`, a standard-library HTTP server that serves safetensors bytes and honours `Range` requests, with a configurable bandwidth throttle. The client connects over loopback. This is an explicit **stand-in** for MinIO/S3 plus kernel traffic shaping (`tc`). It preserves the protocol-level behaviour under test — byte-range retrieval, request/response latency, and a bandwidth ceiling — but it does not reproduce production network jitter, cross-AZ transfer costs, or S3 request-level throttling.

### 7.3 Metrics

| Metric | Definition | Instrumentation |
|---|---|---|
| **TTFT** | Invocation start → first token emitted | Wall clock in `PipelineDriver` |
| **End-to-end** | Invocation start → last layer complete | Wall clock |
| **I/O masking ratio** | $M$ from Section 5.3 | `io_wait_s` vs. `compute_s` in driver metrics |
| **Peak RSS** | Maximum resident set size | `psutil` sampling in harness |
| **RSS trajectory** | RSS over time | `psutil` time series |

### 7.4 Procedure

1. Run `setup.sh` to install dependencies and generate the synthetic checkpoint.
2. For each mode, run $N$ repetitions after a discarded warm-up, to separate cold-path from warm-path effects.
3. Record per-run TTFT, end-to-end latency, masking ratio, and peak RSS into CSV.
4. Report mean, median, p50, and p95 across repetitions; percentiles matter more than means for cold-start claims.
5. Report the RSS trajectory for at least one run per mode.

### 7.5 Threats to validity

- **Loopback ≠ real S3.** Loopback has negligible jitter and no cross-region transfer; real object storage adds tail latency and request throttling. Absolute latencies will not transfer.
- **Simulated compute ≠ transformer.** The synthetic kernel is deterministic and calibrated to be a stand-in, not a faithful GEMM/attention implementation. Masking ratios depend on the ratio $C/D$, which the synthetic kernel only approximates.
- **Single node.** Client, server, and storage share one machine; CPU contention between the HTTP server and the compute driver can distort overlap.
- **No `tc`.** Bandwidth is throttled in-process by the server, not by the kernel; scheduling effects differ from real traffic shaping.
- **Cold start not modelled.** Worker provisioning ($T_{\text{startup}}$) is not exercised; the harness measures the data path from an already-running process. $T_{\text{startup}}$ is therefore excluded from measured TTFT and appears only in the model.
- **Pre-allocation.** The naive implementation holds all parameters, so peak-RSS comparisons cannot demonstrate streaming savings (Section 9.3).

Because of these threats, the results support **relative, protocol-level** conclusions (does overlap reduce TTFT? by roughly how much?) rather than absolute production latency claims.

---

## 8. Results

> **Integrity note.** These are measured values, not estimates. They were produced on 2026-11-11 by `bash setup.sh && bash run.sh` and written to `results/benchmark.csv`. The workload is the synthetic 8-layer F32 checkpoint (8.00 MiB), served by the local RangeServer over loopback with an 8 Mbit/s shared throttle, lookahead 2, work-factor 1.0, three runs per mode, each in a fresh spawned process for RSS isolation. **TTFT here means completion of the full forward pass** (a real first token requires every layer); `first_layer_s` is reported separately as streaming-fill latency and is not an LLM TTFT.

### 8.1 Primary comparison

| Run | Mode | TTFT (s) | End-to-end (s) | Masking ratio $M$ | Peak RSS (MB) |
|---|---|---|---|---|---|
| R1 | Monolithic | 15.4816 | 15.4816 | 0.0000 | 53.688 |
| R1 | Pipelined | 11.0237 | 11.0237 | 0.6262 | 47.902 |
| R2 | Monolithic | 15.6011 | 15.6011 | 0.0000 | 53.738 |
| R2 | Pipelined | 10.8122 | 10.8122 | 0.6263 | 48.777 |
| R3 | Monolithic | 16.0240 | 16.0240 | 0.0000 | 53.574 |
| R3 | Pipelined | 10.6129 | 10.6129 | 0.6258 | 48.172 |
| Mean | Monolithic | 15.7022 | 15.7022 | 0.0000 | 53.667 |
| Mean | Pipelined | **10.8163** | **10.8163** | **0.6261** | 48.284 |
| p50 | Monolithic | 15.6011 | 15.6011 | 0.0000 | 53.688 |
| p50 | Pipelined | 10.8122 | 10.8122 | 0.6263 | 48.172 |
| p95 | Monolithic | 15.9817 | 15.9817 | 0.0000 | 53.730 |
| p95 | Pipelined | 11.0026 | 11.0026 | 0.6263 | 48.720 |

Mean TTFT falls from **15.70 s to 10.82 s** — a **31.1% reduction**, speedup **1.45×**. Mean masking is **0.626 (62.6%)**, below the 70% guardrail and consistent with the layer-0-exposure ceiling $1-d_0/D$. The separate streaming-fill metric (`first_layer_s`) is 9.58 s baseline vs 3.96 s pipeline; it is useful for pipeline diagnosis but is **not** the LLM TTFT.

### 8.2 RSS trajectory (run 1)

| Time slice | Monolithic RSS (MB) | Pipelined RSS (MB) |
|---|---|---|
| $t_0$ (start) | 43.25 | 43.29 |
| $t_1$ ($\approx T/3$) | 48.57 | 47.21 |
| $t_2$ ($\approx 2T/3$) | 53.68 | 47.90 |
| $t_3$ (first token) | 53.69 | 47.90 |
| Peak | 53.69 | 47.90 |

The pipeline's peak is lower here because the baseline holds the entire 8 MiB body while the pipeline holds a bounded window. This is **not** a transformer peak-RSS claim: the synthetic baseline is not a pre-allocated parameter store, and the defensible memory statement remains "bounded transient buffer, no claimed parameter-memory saving."

### 8.3 Reproduction

```bash
bash setup.sh          # idempotent deps + synthetic models
bash run.sh            # 3 runs/mode -> results/benchmark.csv
```

Per-layer overlap detail is in `results/per_layer.csv`; RSS traces are in `results/rss/`. In the pipeline runs, layer 0 is exposed (fetch ≈3.13 s), while layers 1–7 have `blocked_s` ≈ 0.00002–0.00008 s and fetch times ≈1.1–1.8 s under compute — the mechanism working as modelled.

---

## 9. Discussion

### 9.1 Measured trends

The measured run supports the central mechanism claim:

- **TTFT falls from 15.70 s to 10.82 s** (mean, 31.1% reduction, 1.45×). This is below the ~2× ceiling, as the model predicts, because the checkpoint still crosses the link and every layer still computes.
- **Masking is 62.6%**, not 70%+. The exposed time is almost exactly layer 0's 3.13 s download; layers 1–7 are fully hidden (`blocked_s` ≈ 0.00005 s). The measured value is therefore close to the geometric ceiling $1-d_0/D$ for this workload.
- **End-to-end and TTFT coincide** because the first token requires the full forward pass in an autoregressive decoder. The streaming-fill metric (`first_layer_s`) drops from 9.58 s to 3.96 s but is not an LLM TTFT.
- **Peak RSS was 53.7 MB baseline vs 48.3 MB pipelined** in this synthetic harness. We do **not** claim this as a transformer memory saving: the baseline holds the whole 8 MiB file, while the pipeline holds a bounded window, and neither pre-allocates a full parameter store.

### 9.2 Why masking is not 100% on CPU

The model in Section 5.4 bounds masking by available compute. In the measured run, the only unavoidable exposure is layer 0: its 3.13 s download happens before any compute can start. That alone caps $M$ at $1-d_0/D \approx 0.63$. All subsequent layer downloads fit under the preceding layer's compute, so the pipeline achieves essentially that ceiling. The earlier 30–60% expectation was too pessimistic for this configuration because per-layer compute (≈1 s) is comparable to per-layer download (≈1.05 s at 8 Mbit/s over 1 MiB). A faster compute backend or a slower link would move the system toward the network-bound regime and change $M$; that is an empirical question for the next run, not a claim we make now.

### 9.3 Why no memory claim

The defensible wins of this artifact are **TTFT** and **I/O masking**. The peak-RSS win is *not* claimed, because the driver still materialises the full parameter set: the ring buffer bounds prefetched payloads, not the parameter store. A true streaming design requires a custom parameter store (mmap-backed arena, or per-layer tensors freed after use) or quantisation. We state this plainly rather than reporting a misleading RSS delta that would only reflect buffer depth.

### 9.4 What would change the conclusions

- A GPU backend would make compute much faster relative to transfer, likely *lowering* $M$ (network becomes the bottleneck) but *increasing* the relative TTFT benefit of overlap.
- Real S3/MinIO with `tc` would add tail latency, likely lowering $M$ and increasing variance.
- A larger model would increase $D$ and the number of layers, giving the pipeline more opportunity to overlap and making the TTFT benefit more visible.
- A custom parameter store would convert the memory story from "bounded buffer" to "bounded working set", enabling a genuine peak-RSS claim.

---

## 10. Publication Viability

### 10.1 Realistic framing

This is an Experiential Learning project, not a completed research programme. Its contribution is a working, honestly-evaluated prototype plus a clean problem formulation. The realistic publication path is:

1. **Now:** EL submission and viva (this document).
2. **Next:** a workshop paper or a short paper at an Indian IEEE conference, after replacing the local Range server with MinIO + `tc`, adding measured results, and evaluating at least one frozen public checkpoint (SmolLM or TinyLlama).
3. **Later:** a full paper only if the memory story is closed with a custom parameter store and the evaluation covers multi-tenant cold starts.

### 10.2 Target venues

| Venue | Type | Fit |
|---|---|---|
| **IEEE CloudCom** | International conference, cloud computing | Strong fit: serverless cold start + object storage |
| **IEEE INDICON** | India-wide IEEE conference | Good fit for a systems prototype with measured results |
| **IEEE DISCOVER** | IEEE conference, computing/communication | Reasonable fit; broad scope |
| **IEEE CONFLUENCE** | IEEE conference, emerging technologies | Reasonable fit; suitable for a short paper |

### 10.3 Distinct contribution vs. the five works

| Work | What it does | What this project adds | Overlap / limitation |
|---|---|---|---|
| ServerlessLLM [1] | GPU checkpoint loading, migration | Layer-granular Range streaming on a CPU worker | Shares cold-start motivation; weaker hardware story |
| LLM in a Flash [2] | Flash weight streaming, neuron prediction | Remote object storage + explicit manifest | Shares I/O–compute overlap; no predictor |
| vLLM [3] | Paged KV cache, batching | Cold-start data path *before* serving | Complementary, not competing |
| FastServe [4] | Request scheduling, preemption | Weight availability for the first request | Complementary, not competing |
| SOCK [5] | Container dependency prefetch | Model-internal weight streaming | Different granularity; shares cold-path framing |

**Novelty caveat.** Each individual mechanism — safetensors parsing, HTTP Range, prefetching, ring buffers — is well known. The contribution is their *combination* at the serverless cold-start boundary and the empirical accounting of how much overlap a CPU worker actually achieves. That is a modest, defensible claim. It is not a new scheduler, a new memory manager, or a new model architecture, and the paper should not imply otherwise.

### 10.4 Minimum evidence for submission

- Real MinIO (or S3) backend with kernel-level `tc` throttling.
- Measured TTFT, masking, and RSS across $N$ repetitions with p50/p95.
- At least one frozen public checkpoint, not only the synthetic model.
- A stated, measured value of $\eta$ and an explanation of the gap to $M_{\text{ideal}}$.

---

## 11. Conclusion and Future Work

### 11.1 Conclusion

We designed and implemented an asynchronous progressive layer-streaming engine for serverless LLM invocations. The engine parses safetensors byte ranges, emits a layer manifest, fetches layers over HTTP Range with an asyncio prefetcher, bounds the working set with a ring buffer and RSS clamp, and drives a layer-by-layer compute loop. A benchmark harness compares it against a monolithic baseline. The design is grounded in a latency model that separates startup, download, compute, and residual stall, and in an explicit masking-ratio definition with an Amdahl-like bound.

The honest summary is narrow but real: the artifact demonstrates protocol-level layer streaming and overlap accounting, and the measured run shows a 1.45× TTFT improvement with 62.6% masking. It does not yet demonstrate peak-RSS reduction, it does not run a real transformer, and it does not exercise a production serverless cold start. Those are stated as limitations, not hidden. All quantitative results come from `results/benchmark.csv`; no number in this report is invented.

### 11.2 Future work

1. **Custom parameter store.** Replace full pre-allocation with an mmap-backed arena or per-layer allocation/free to obtain a genuine peak-RSS reduction, closing the gap between $R_{\text{naive}}$ and $R_{\text{stream}}$.
2. **Real MinIO + `tc`.** Swap the local Range server for MinIO and shape traffic with `tc` to measure production-like tail latency.
3. **GPU backend.** Move the synthetic kernel to a GPU to study the regime where compute is fast and the network dominates.
4. **Frozen model evaluation.** Evaluate SmolLM and TinyLlama checkpoints, reporting latency only after measurement.
5. **Cold-start integration.** Model and measure $T_{\text{startup}}$ by deploying the worker as an actual serverless function.
6. **Multi-tenant scheduling.** Study how concurrent cold starts contend for object-storage bandwidth.
7. **Compression and quantisation.** Reduce $D$ and $P$ simultaneously via INT8/INT4 or compressed weight formats.
8. **Intra-layer chunking.** Implement sub-layer chunks so that $s^{\star} = c_i \cdot BW$ can be realised when a layer is larger than the compute window.

---

## References

[1] Y. Fu, L. Xue, Y. Huang, A.-O. Brabete, D. Ustiugov, Y. Zhang, and L. Mai, "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models," in *Proc. 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, 2024.

[2] K. Alizadeh, I. Mirzadeh, D. Belenko, K. Khatamifard, M. Cho, C. C. Del Mundo, M. Rastegari, and M. Farajtabar, "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory," arXiv:2312.11514, Dec. 2023.

[3] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proc. 29th ACM Symposium on Operating Systems Principles (SOSP '23)*, 2023.

[4] B. Wu, Y. Zhong, Z. Zhang, G. Huang, X. Liu, and X. Wang, "Fast Distributed Inference Serving for Large Language Models," arXiv:2305.05920, May 2023.

[5] A. Oakes, L. Yang, D. Zhou, K. Houck, T. Harter, A. C. Arpaci-Dusseau, and R. H. Arpaci-Dusseau, "SOCK: Rapid Task Provisioning with Serverless-Optimized Containers," in *Proc. USENIX Annual Technical Conference (ATC '18)*, 2018.

[6] J. M. Hellerstein, J. Faleiro, J. E. Gonzalez, J. Schleier-Smith, V. Sreekanti, A. Tumanov, and C. Wu, "Serverless Computing: One Step Forward, Two Steps Back," in *Proc. 9th Biennial Conference on Innovative Data Systems Research (CIDR '19)*, 2019.

[7] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.

---

*Document status: report complete; measured results were produced by `bash run.sh` on 2026-11-11 and are recorded in `results/benchmark.csv`. No benchmark number has been fabricated.*
