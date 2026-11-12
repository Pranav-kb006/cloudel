STATUS: Viva Q&A COMPLETE — 5 tough questions, deployment reality-check box, 3 rapid-fire questions, elevator pitch, and "three things" summary. Quantitative answers use the measured 2026-11-11 benchmark (`results/benchmark.csv`): TTFT 15.70 s → 10.82 s (1.45×), masking 0.626, first-layer fill 9.58 s → 3.96 s. No numbers are fabricated. No Python packages are needed for these docs.

---

# Viva Voce Preparation — Async Progressive Layer-Streaming Engine

**Project:** Design and Empirical Evaluation of an Asynchronous Progressive Layer-Streaming Engine for Accelerating Serverless Large Language Model Invocations
**Course:** IS355TBD (Cloud Computing), 5th Sem, RVCE — EL Project
**Date:** 2026-11-11
**Team:** ____

> **How to use this document.** Each answer is written to be said aloud in 60–90 seconds. Numbers come from the measured 2026-11-11 benchmark (`results/benchmark.csv`: 3 runs/mode, synthetic 8-layer model, 8 Mbit/s shared throttle). If an examiner asks for a number we did not measure, say so and explain what would produce it. Examiners reward calibrated honesty far more than invented precision.

---

## Status header

| Item | State |
|---|---|
| Artifact | Built and exercised: parser, manifest writer, prefetcher, buffer, driver, Range server, harness (`bash setup.sh && bash run.sh`) |
| Measured results | **Measured 2026-11-11**: TTFT 15.70 s → 10.82 s (1.45×), masking 0.626, first-layer fill 9.58 s → 3.96 s |
| Compute backend | Deterministic synthetic CPU stand-in (no torch/numpy) |
| Storage backend | Local stdlib Range server over loopback (stand-in for MinIO/S3) |
| Real LLM forward pass | Not claimed; frozen SmolLM/TinyLlama is future work |
| Peak-RSS saving | **Not claimed** in the naive implementation |
| Masking ratio | **Measured 62.6%** (below 70%; close to the $1-d_0/D$ ceiling) |

---

## Q1. "Doesn't async prefetch fully hide I/O? What is your actual masking ratio?"

**Short answer:** No — it does not fully hide I/O, and the measured masking ratio is **0.626 (62.6%)**, not 70%+. The exposed time is almost entirely layer 0's download, which cannot overlap any compute because it must arrive first.

**Full answer:**

The ideal model says at most $C$ seconds of transfer can hide behind $C$ seconds of compute, so

$$
M \le M_{\text{ideal}} = \min\!\left(1, \frac{C}{D}\right),
$$

and the masking ratio is

$$
M = \frac{D - d_0 - \sum_{i=1}^{L-1}\max(0,\, d_i - c_{i-1})}{D}.
$$

Because layer 0 is exposed by construction, a tighter ceiling for this pipeline is

$$
M \le 1 - \frac{d_0}{D}.
$$

In the measured run, $D \approx 8.39$ s at 8 Mbit/s and $d_0 \approx 3.13$ s, so the ceiling is $1 - 3.13/8.39 \approx 0.627$. We measured **0.626**. Every layer after layer 0 was already buffered when compute needed it (`blocked_s` ≈ 0.00005 s). So the honest statement is not "prefetch hides everything"; it is "prefetch hides everything except the unavoidable first-layer fill."

That still gives a real TTFT win: **15.70 s baseline → 10.82 s pipelined** (1.45×, 31.1% lower), where TTFT means completion of the full forward pass — a real first token needs every layer. The separate `first_layer_s` metric falls from 9.58 s to 3.96 s, but that is streaming-fill latency, not an LLM TTFT.

---

## Q2. "If you still pre-allocate all parameters, where is the memory saving?"

**Short answer:** In the naive implementation there is **no peak-RSS saving**. The defensible wins are TTFT and I/O masking.

**Full answer:**

The memory trajectory of the current driver is

$$
R_{\text{naive}}(t) = R_0 + P + \sum_{j \in W(t)} s_j,
\qquad P = \sum_{i=1}^{L} s_i,
$$

where $P$ is the full parameter set and $W(t)$ is the set of layers currently buffered. Because $P$ is allocated up front, peak RSS is dominated by $P$ in both monolithic and pipelined modes. The ring buffer only bounds the *transient* prefetched payloads:

$$
B_{\max} \le N_{\text{look}} \cdot \max_i s_i.
$$

So we deliberately do **not** claim a peak-RSS reduction. The measured harness shows peak RSS 53.7 MB baseline vs 48.3 MB pipelined, but that gap reflects the baseline holding the whole 8 MiB file while the pipeline holds a bounded window. In a real transformer the pre-allocated parameter store $P$ would dominate both modes, so the measured gap would not survive. A true streaming implementation would drop the $P$ term:

$$
R_{\text{stream}}(t) = R_0 + \sum_{j \in W(t)} s_j,
$$

which requires a custom parameter store — an mmap-backed tensor arena, or per-layer allocation and free after use — or quantisation. We list that as future work. Claiming a memory win today would be dishonest, and we would rather defend a narrow true claim than a broad false one.

**If pressed "then why build the buffer at all?"** Because it is the mechanism that *would* deliver the saving once the parameter store is replaced, and it already prevents unbounded prefetch from inflating RSS beyond the clamp. It is infrastructure for the next step, not a finished memory optimisation.

---

## Q3. "Why safetensors and HTTP Range instead of just downloading the file / using mmap?"

**Short answer:** Random access, no pickle, exact byte ranges from the header, and mmap does not cross a network boundary.

**Full answer:**

Three reasons.

1. **The format is self-describing and random-access by construction.** A safetensors file is `[8-byte little-endian header length][JSON header][data section]`. The JSON header gives every tensor's `dtype`, `shape`, and `data_offsets`. We parse the 8-byte length, parse the JSON, and compute absolute byte offsets. No scanning, no guessing.
2. **No arbitrary code execution.** Pickle-based checkpoints (`torch.save`) deserialise by executing code; safetensors is pure data. For a multi-tenant serverless platform that is a real security property, not a nicety.
3. **HTTP Range makes layer granularity a network primitive.** With a manifest mapping layer → object key + `[start, end)`, the worker fetches exactly one layer. Downloading the whole file defeats the purpose; it is the monolithic baseline we are trying to beat.

**Why not mmap?** `mmap` is excellent *locally* — it gives demand paging over a file on disk. But in serverless the checkpoint is in object storage, not on the worker's local filesystem. You cannot `mmap` an S3 object. You could `mmap` a local cache file, but that presupposes the file is already downloaded — which is the cost we are removing. So mmap is a valid *local-tier* optimisation and a non-starter as the *remote-tier* mechanism.

**If pressed "isn't Range just as expensive as a full GET?"** Per byte it is comparable, but you do not pay for bytes you have not requested yet, and you can start computing on layer 1 while layers 2..L are still in flight. The win is scheduling, not raw throughput.

---

## Q4. "How is this different from ServerlessLLM / LLM in a Flash / vLLM / FastServe / SOCK?"

**Short answer:** Each solves a different layer of the stack; we combine layer-granular remote streaming with async overlap and RSS-aware buffering at the serverless cold-start boundary.

**Full answer — distinct-contribution comparison:**

| Work | Solves | Granularity | Our difference |
|---|---|---|---|
| **ServerlessLLM** (OSDI '24) | Checkpoint loading into GPU memory, live migration | Whole checkpoint / model | We stream individual layers to a CPU worker over Range; no GPU copy |
| **LLM in a Flash** (arXiv 2023) | Weights exceed device DRAM; stream from flash | Neuron / row, with predictor | We target remote object storage with a static manifest, not local flash |
| **vLLM / PagedAttention** (SOSP '23) | KV-cache fragmentation, throughput | KV blocks / sequences | We optimise the cold path *before* serving; vLLM assumes weights are resident |
| **FastServe** (arXiv 2023) | Queueing delay via skip-join MLFQ scheduling | Request / token | We make the first request's weights available sooner; complementary to scheduling |
| **SOCK** (ATC '18) | Container provisioning and dependency prefetch | Container image / package | We stream model-internal weights; SOCK stops at the process boundary |

The combination we claim is: layer-granular random access to a safetensors checkpoint **+** remote object-storage retrieval via HTTP Range **+** async prefetch/compute overlap **+** RSS-aware bounded buffering, aimed at serverless TTFT. No single one of the five sits at that intersection.

**Admitted overlaps and limitations:**

- We share the cold-start motivation with ServerlessLLM and SOCK.
- We share the I/O–compute overlap idea with LLM in a Flash; we lack its predictor.
- We are complementary to vLLM and FastServe rather than competitive.
- Our evidence is a CPU prototype with synthetic compute; the others have production-scale GPU evaluations. We do not claim parity.

**If pressed "so what is genuinely new?"** The *empirical accounting* of how much overlap a single-process CPU worker actually achieves, and the explicit decomposition into $M_{\text{ideal}}$ and $\eta$. That is a modest but real contribution, and we frame it as an EL project with a path to a short paper — not a replacement for any of the five.

---

## Q5. "Is your benchmark a real serverless deployment?"

**Short answer:** No. It is a local Range server over loopback, a stand-in for MinIO/S3 plus `tc`, with no cold-start or multi-tenant effects.

**Full answer:**

The benchmark runs three processes on one machine: the client/driver, a stdlib HTTP Range server, and the synthetic model generator. Storage is a local directory served over loopback with an in-process bandwidth throttle. That is **not** a serverless deployment. Specifically, it does not reproduce:

- worker provisioning and container pull ($T_{\text{startup}}$ in our model);
- network jitter, cross-AZ transfer, or S3 request throttling;
- kernel-level traffic shaping (`tc`);
- multi-tenant contention for egress bandwidth;
- the real cold-start path of Lambda / Cloud Run / OpenFaaS.

What it *does* preserve is the protocol-level behaviour under test: HTTP Range retrieval of a layer, request/response latency, a bandwidth ceiling, and the scheduling relationship between transfer and compute. So the results support **relative** conclusions — does overlap reduce TTFT, and roughly by how much — not absolute production latency claims. The production path is MinIO + `tc` + a real worker, and we state that explicitly in the report.

**If pressed "then is the result meaningful at all?"** Yes, at the level of mechanism validation: it shows the engine issues correct range requests, overlaps them with compute, and accounts for the overlap. It does not validate a production SLA, and we do not present it as doing so.

---

## Deployment vs. serverless reality check

> **Box: what this is, and what it is not**
>
> | Aspect | This evaluation | Production serverless |
> |---|---|---|
> | Worker | Local Python process | Lambda / Cloud Run / OpenFaaS worker |
> | Storage | Local stdlib Range server | S3 / MinIO / GCS |
> | Network | Loopback | Cross-AZ / cross-region, jittery |
> | Bandwidth control | In-process throttle | Kernel `tc` / provider egress limits |
> | Cold start ($T_{\text{startup}}$) | Not exercised | Dominant term |
> | Tenancy | Single tenant | Multi-tenant, contended |
> | Compute | Synthetic CPU kernel | GPU transformer forward |
>
> **Local process ≠ Lambda/Cloud Run. Loopback bandwidth ≠ S3. Throttle is simulated, not shaped.**
> The production path is MinIO + `tc` + a real worker; that is future work, not a current claim.

---

## Rapid-fire questions

**R1. Why layer granularity rather than tensor or neuron granularity?**
Because a layer is the unit of execution: the driver loads, binds, computes, and evicts one layer at a time. Tensor granularity would add request overhead without enabling earlier compute; neuron granularity (as in LLM in a Flash) needs a predictor we do not have. Layer granularity is the coarsest unit that still overlaps.

**R2. What exactly is TTFT?**
Time-to-first-token: wall-clock from invocation start to the emission of the first output token. In our model it is $S + D + C$ for the monolithic path and $S + d_1 + C + \text{residual stalls}$ for the pipelined path. It is the metric that matters for interactive use, because users perceive the first token, not the last.

**R3. What happens on a short read or a failed Range request?**
The prefetcher must treat the response as incomplete and either retry the missing range or fail the layer. The ring buffer must not publish a partial payload to the driver, otherwise the compute step would bind truncated parameters. The exact retry policy belongs in `src/core/AsyncPrefetcher.py`; this document does not assert a specific retry count. Correctness requires that a layer is only marked ready when its full `[start, end)` range has arrived.

---

## Elevator pitch (one line)

> We let a serverless LLM worker start computing on the first layer of a safetensors checkpoint while later layers are still streaming from object storage, trading a bounded buffer for lower time-to-first-token — and we measure honestly how much of the I/O that actually hides.

---

## If you only remember three things

1. **The win is TTFT and I/O masking, not memory.** The naive implementation pre-allocates all parameters, so peak RSS is not reduced; a custom parameter store is future work.
2. **Masking on CPU measured 62.6%, not 70%+.** The exposed time is layer 0's unavoidable download; later layers were fully hidden. Do not generalise this to a real transformer without measuring it.
3. **This is a local stand-in, not a serverless deployment.** Range server + loopback replaces MinIO/S3 + `tc`; results are protocol-level and relative, and no production latency is claimed.

---

*Document status: viva preparation complete; quantitative answers use the measured 2026-11-11 benchmark in `results/benchmark.csv`. No fabricated numbers.*
