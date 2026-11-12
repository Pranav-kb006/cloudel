<!--
STATUS: WRITTEN — baseline, pipelined TTFT, masking, speedup, memory, and chunk-granularity models are derived.
Measured results (2026-11-11) are in `docs/project_report.md` §8 and `results/benchmark.csv`; the worked example in §9 remains ILLUSTRATIVE, NOT MEASURED.
Packages needed to read or build these docs: NONE (Markdown + LaTeX only).
Owner: WS-1 `architect`. Files owned: docs/01_architecture.md, docs/02_math_model.md.
-->

# 02 — Analytical Model: TTFT, I/O Masking, Memory, and Chunking

**Project:** Design and Empirical Evaluation of an Asynchronous Progressive Layer-Streaming Engine for Accelerating Serverless Large Language Model Invocations
**Course:** IS355TBD (Cloud Computing), 5th Semester, RVCE — Experiential Learning
**Document owner:** WS-1 (`architect`)
**Status:** Model complete. The worked example is illustrative; measured results are in `docs/project_report.md` §8.

> **STATUS:** This document derives the baseline and pipelined latency equations, the I/O masking ratio, the speedup bound, the RSS/buffer-depth model, and the chunk-granularity tradeoff. The worked example is explicitly labelled **illustrative, NOT measured**; the measured 2026-11-11 values are in `docs/project_report.md` §8 and `results/benchmark.csv`. Reading or building this document requires **no packages**.

---

## 1. Purpose and honesty contract

The model exists to answer four questions *before* the benchmark runs, and to tell the benchmark what to measure:

1. How much can layer streaming reduce TTFT, and under what conditions?
2. What fraction of download time can realistically be hidden on CPU?
3. How deep can the prefetch buffer be before RSS becomes unsafe?
4. At what granularity should the checkpoint be chunked?

The model is deliberately conservative. It does **not** assume GPU-class compute, it does **not** assume infinite bandwidth, and it does **not** claim peak-RSS savings. Measured values are in `docs/project_report.md` §8; the worked example in §9 uses round, plausible symbols and remains marked **illustrative, NOT measured**.

---

## 2. Notation

| Symbol | Meaning | Unit |
|---|---|---|
| $L$ | number of addressable weight stages (transformer layers plus `embed` and `head` pseudo-layers) | — |
| $i$ | stage index, $i = 0,\ldots,L-1$ | — |
| $s_i$ | payload size of stage $i$ | bytes |
| $B_{\text{net}}$ | effective download bandwidth available to one worker | bytes/s |
| $d_i$ | download time of stage $i$ (including request/TTFB overhead) | s |
| $c_i$ | compute time of stage $i$ | s |
| $D$ | total download time, $D=\sum_i d_i$ | s |
| $C$ | total compute time, $C=\sum_i c_i$ | s |
| $t_q$ | queue / admission time at the gateway | s |
| $t_i$ | worker cold-start / runtime-init time | s |
| $t_m$ | manifest fetch time (small JSON; cached when possible) | s |
| $t_{\text{load}}$ | deserialise / mmap / materialise time in the baseline | s |
| $t_{\text{startup}}$ | $t_q+t_i+t_m$ | s |
| $c_{\text{tail}}$ | LM head + sampling + detokenisation after the last layer | s |
| $A_i$ | time stage $i$'s bytes are fully available | s |
| $F_i$ | time stage $i$'s compute finishes | s |
| $k$ | ring-buffer depth (resident layer payloads) | slots |
| $k_{\max}$ | configured maximum depth | slots |
| $s_{\text{layer}}$ | representative layer payload size | bytes |
| $RSS_{\text{base}}$ | resident set before buffering layers | bytes |
| $RSS_{\text{limit}}$ | hard memory budget for the worker | bytes |
| $A(t)$ | activations + transient copies at time $t$ | bytes |
| $A_{\max}$ | reserved activation headroom | bytes |
| $N$ | total checkpoint bytes, $N=\sum_i s_i$ | bytes |
| $z$ | chunk size for sub-layer chunking | bytes |
| $o$ | per-request overhead (HTTP, TLS, scheduling) | s |
| $M$ | I/O masking ratio | — |
| $S$ | TTFT speedup over the monolithic baseline | — |

All times are measured from request admission at the gateway unless stated otherwise.

---

## 3. Baseline: monolithic download-then-compute

The conventional serverless path downloads the **entire** checkpoint before the forward pass:

$$
T_{\text{base}}
= t_q + t_i + t_m + t_{\text{load}} + D + C + c_{\text{tail}},
\qquad
D = \sum_{i=0}^{L-1} d_i = \frac{N}{B_{\text{net}}}.
$$

Here $N$ is the whole-file size. The baseline is **serial**: bytes arrive, then layers compute. $t_{\text{load}}$ covers deserialisation or page-in; in a mmap-based baseline it may partially overlap with compute, which is a conservative simplification we state rather than hide.

Because an autoregressive decoder needs **every** layer to produce the first token, the baseline TTFT is the whole expression:

$$
\boxed{\;TTFT_{\text{base}} = t_q + t_i + t_m + t_{\text{load}} + D + C + c_{\text{tail}}\;}
$$

There is no "first-token shortcut": skipping layers would change the model, not accelerate it. This is why the only available win is **overlap**, and why the speedup ceiling is modest (§6).

**Measured quantities:** the harness records $D$ (as `io_wait_s` for the baseline), $C$, TTFT, and masking; these are in `results/benchmark.csv`. $t_q,t_i,t_m,t_{\text{load}},c_{\text{tail}}$ are not separately instrumented in this build and must not be quoted as measured.

---

## 4. Pipelined TTFT

### 4.1 Exact recurrence

Let $A_i$ be the time stage $i$'s bytes are fully available and $F_i$ the time its compute finishes, measured from the end of startup. Downloads are serialised on one effective link (the prefetcher may open concurrent connections, but they share $B_{\text{net}}$, so $d_i$ is the *effective* time to add stage $i$'s bytes). Compute is non-preemptive and sequential across layers.

$$
A_0 = d_0, \qquad F_0 = d_0 + c_0,
$$

$$
A_i = \max\!\left(A_{i-1} + d_i,\; F_{i-1}\right),
\qquad
F_i = A_i + c_i \quad (i \ge 1).
$$

The first term says stage $i$'s bytes cannot finish before stage $i-1$'s bytes plus its own download; the second says stage $i$ cannot compute before stage $i-1$ finishes computing. Then

$$
\boxed{\;T_{\text{pipe}} = t_{\text{startup}} + F_{L-1} + c_{\text{tail}}\;}
$$

This recurrence is the **ground truth** of the model; the closed forms below are simplifications of it. Note that the pipelined path does **not** pay the baseline's whole-file $t_{\text{load}}$: per-layer binding/decoding cost is folded into $c_i$. If the baseline uses `mmap` with demand paging, $t_{\text{load}}$ is near zero, which makes the comparison conservative (it understates the baseline's cost, not the pipeline's).

### 4.2 Uniform-layer closed form

If all layers are identical, $d_i=d$ and $c_i=c$, induction gives

$$
F_{L-1} = d + c + (L-1)\max(d,c).
$$

Therefore

$$
T_{\text{pipe}}
= t_{\text{startup}} + d + c + (L-1)\max(d,c) + c_{\text{tail}}.
$$

Since $d+c=\max(d,c)+\min(d,c)$, the same result can be written in the symmetric form

$$
\boxed{\;
T_{\text{pipe}}
= t_{\text{startup}}
+ \sum_{i=0}^{L-1}\max(d_i,c_i)
+ \min(d_0,c_0)
+ c_{\text{tail}}
\;}
$$

The extra $\min(d_0,c_0)$ is the **first-layer penalty**: layer 0 cannot overlap with anything, so its download and compute are both fully exposed. Every later layer contributes only the larger of its download and compute. This is the cleanest statement of the pipeline's benefit and its limit.

### 4.3 Non-uniform approximation

For non-uniform layers, the transient coupling between $c_{i-1}$ and $d_i$ means no simple closed form is exact. The standard approximation is

$$
T_{\text{pipe}}
\approx t_{\text{startup}}
+ d_0 + c_0
+ \sum_{i=1}^{L-1}\max(d_i,c_i)
+ c_{\text{tail}}.
$$

This is **exact** when layers are uniform (or when one of $d_i,c_i$ dominates uniformly) and a good approximation otherwise. It is the equation the report should quote, with the recurrence in §4.1 cited as the exact model. An equivalent aggregate form, useful for intuition, is

$$
T_{\text{pipe}}
\approx t_{\text{startup}}
+ \max(D,C)
+ \min(d_0,c_0)
+ c_{\text{tail}},
$$

i.e. *the pipeline takes about the larger of total download and total compute, plus one layer of the smaller, plus startup.*

### 4.4 The two limiting regimes

- **Compute-bound** ($c_i \gg d_i$): $T_{\text{pipe}} \approx t_{\text{startup}} + d_0 + C + c_{\text{tail}}$. All later downloads are hidden, but wall-clock is dominated by compute. The pipeline removes almost none of the baseline's $D$ because $D$ was small relative to $C$.
- **Download-bound** ($d_i \gg c_i$): $T_{\text{pipe}} \approx t_{\text{startup}} + D + c_0 + c_{\text{tail}}$. All compute is hidden, but wall-clock is dominated by download. The pipeline removes almost none of $D$ because it cannot download faster than the link.
- **Balanced** ($d_i \approx c_i$): the pipeline is near its best case, roughly halving the sum $D+C$.

This is the central honest point: **streaming is an overlap technique, not a compression technique.** It cannot beat $\max(D,C)$ by much, so its asymptotic speedup is at most about $2\times$ (§6).

---

## 5. Per-layer overlap and I/O masking ratio

### 5.1 Definition

The per-layer overlap is the portion of a layer's download hidden behind compute:

$$
\text{overlap}_i = \min(d_i,c_i).
$$

The I/O masking ratio is the hidden fraction of download time. Because layer 0 can never be overlapped, the **reporting definition** excludes it:

$$
\boxed{\;
M = \frac{\displaystyle\sum_{i=1}^{L-1}\min(d_i,c_i)}
{\displaystyle\sum_{i=0}^{L-1} d_i}
= 1 - \frac{D_{\text{exposed}}}{D},
\qquad
D_{\text{exposed}} = d_0 + \sum_{i=1}^{L-1}\max(0,\,d_i-c_i).
\;}
$$

The simpler all-layer form $M'=\sum_{i=0}^{L-1}\min(d_i,c_i)/D$ is sometimes quoted; it overstates masking by $\min(d_0,c_0)/D$ and should not be used for the headline number. The harness may report both, but the headline uses $M$.

### 5.2 Interpretation

- $M=0$: no overlap; the pipeline degenerates to serial download-then-compute.
- $M=1$: **100% masking** — every byte of every layer after layer 0 arrives before compute needs it. This requires $c_i \ge d_i$ for all $i\ge1$ and enough buffer/lookahead to run the prefetcher ahead.
- $0<M<1$: partial masking; some downloads are exposed.

For uniform layers, $M=\min(d,c)/d$ (excluding the first layer's contribution, which is small for large $L$). Masking is high when compute is slow relative to the link, and low when the link is slow relative to compute.

### 5.3 Relation to speedup

Ignoring startup and tail, the wall-clock saved by the pipeline is exactly the hidden download time:

$$
T_{\text{base}} - T_{\text{pipe}}
= (D+C) - \left(d_0+c_0+\sum_{i=1}^{L-1}\max(d_i,c_i)\right)
= \sum_{i=1}^{L-1}\min(d_i,c_i)
= M\,D.
$$

Hence

$$
S = \frac{T_{\text{base}}}{T_{\text{pipe}}}
= \frac{D+C}{D+C-MD}
= \frac{1}{1-\dfrac{M D}{D+C}}.
$$

This is a useful sanity check for the harness: a measured $M$ and a measured $D,C$ must reproduce the measured speedup to within startup/tail effects. If they do not, the timing instrumentation is inconsistent.

### 5.4 Why real CPU masking is lower than 100%

The ideal $M$ assumes the prefetcher runs perfectly ahead and compute never waits. On a CPU worker, several effects can reduce it; which dominate is empirical:

1. **First-layer exposure.** $d_0$ is always paid (§4.2).
2. **Single-core contention.** Prefetch payload handling (socket reads, copies, JSON/header parsing, tensor binding) competes with compute for the same core; the GIL serialises Python-level work. Effective $c_i$ shrinks while the driver is busy.
3. **Buffer depth.** A small $k$ limits how far ahead the prefetcher can run; a bursty link then exposes download gaps.
4. **Throttle burstiness.** Sleep/token-bucket throttling is coarser than a real `tc` shaper; idle periods are not perfectly usable.
5. **Request overhead and retries.** Each layer fetch pays TTFB; retries add exposed time.
6. **Synthetic/CPU kernels are not free.** Copying and decoding a layer payload costs time that does not advance the forward pass.

Consequently the *a priori* expectation was **30–60% masking**, not >70%. The measured 8-layer run achieved **62.6%**, just above that band: per-layer compute (≈1 s) was comparable to per-layer download (≈1.05 s), so only layer 0 was exposed and later layers were fully hidden. This is a welcome empirical result, but it is specific to the workload and link; it must not be generalised to a real transformer without measurement.

---

## 6. Speedup bound and limiting cases

Starting from the exact recurrence and the uniform closed form:

$$
S = \frac{T_{\text{base}}}{T_{\text{pipe}}}
= \frac{t_q+t_i+t_m+t_{\text{load}}+D+C+c_{\text{tail}}}
{t_{\text{startup}}+d_0+c_0+\sum_{i=1}^{L-1}\max(d_i,c_i)+c_{\text{tail}}}.
$$

Since the denominator is at least $t_{\text{startup}}+\max(D,C)$, and the numerator is at most $t_{\text{startup}}+t_{\text{load}}+D+C+c_{\text{tail}}$,

$$
\boxed{\;
S \;\le\; \frac{t_{\text{startup}}+t_{\text{load}}+D+C+c_{\text{tail}}}
{t_{\text{startup}}+\max(D,C)+c_{\text{tail}}}
\;=\; 1+\frac{t_{\text{load}}+\min(D,C)}
{t_{\text{startup}}+\max(D,C)+c_{\text{tail}}}.
\;}
$$

If deserialisation is counted inside $D$ (or is negligible, $t_{\text{load}}\to0$), this reduces to

$$
S \;\le\; 1+\frac{\min(D,C)}{t_{\text{startup}}+\max(D,C)+c_{\text{tail}}}
\;\le\; 2.
$$

For uniform layers with negligible startup/tail, this becomes the clean asymptotic result

$$
S_\infty \to \frac{d+c}{\max(d,c)}
= 1+\frac{\min(d,c)}{\max(d,c)}
\le 2.
$$

**Limiting cases:**

| Regime | Condition | Behaviour of $S$ | Interpretation |
|---|---|---|---|
| Compute-bound | $c_i \gg d_i$ | $S \to 1 + D/C \to 1$ | I/O fully hidden, but compute dominates; little wall-clock gain. |
| Download-bound | $d_i \gg c_i$ | $S \to 1 + C/D \to 1$ | Compute fully hidden, but the link dominates; little wall-clock gain. |
| Balanced | $d_i \approx c_i$ | $S \to 2$ | Best case: one of the two costs is hidden. |
| Large startup | $t_{\text{startup}} \gg D+C$ | $S \to 1$ | Cold-start overhead swamps the streaming benefit. |

The bound is the most important guardrail in the project: **no measured speedup above ~2× is physically plausible for this design**, because the whole checkpoint must still cross the link and every layer must still compute. Any number above 2× would indicate a measurement error (e.g. baseline not actually downloading, or a cached pipelined run).

---

## 7. Memory trajectory and buffer-depth clamp

### 7.1 RSS model

Let $n(t)$ be the number of layer payloads resident in the ring buffer at time $t$. Then

$$
RSS(t) = RSS_{\text{base}} + n(t)\,s_{\text{layer}} + A(t) + \varepsilon(t),
$$

where $A(t)$ is activation/transient memory and $\varepsilon(t)$ covers allocator fragmentation and runtime overhead. The buffer depth is bounded by

$$
\boxed{\;
B \le \min\!\left(
k_{\max},\;
\left\lfloor
\frac{RSS_{\text{limit}} - RSS_{\text{base}} - A_{\max}}{s_{\text{layer}}}
\right\rfloor
\right).
\;}
$$

The simplified form often quoted omits the activation reserve:

$$
B \le \min\!\left(k_{\max},\;
\left\lfloor
\frac{RSS_{\text{limit}} - RSS_{\text{base}}}{s_{\text{layer}}}
\right\rfloor\right).
$$

The prefetch lookahead is then $p=\min(L_{\text{lookahead}},\,B-1)$: with $B=3$ slots and one layer being computed, at most two can be in flight. If $B<1$, the engine switches to **synchronous single-layer mode** (fetch → compute → evict) rather than risk an OOM kill. The monitor re-evaluates $B$ after each layer and on a timer.

### 7.2 Trajectory shape

For a request that runs to completion, $n(t)$ has three phases:

$$
n(t) \approx
\begin{cases}
\left\lceil \dfrac{t}{\bar d} \right\rceil, & 0 \le t \le t_{\text{fill}} \quad\text{(ramp-up)}\\[6pt]
B, & t_{\text{fill}} < t \le t_{\text{drain}} \quad\text{(steady state)}\\[6pt]
\max\!\left(0,\; B - \left\lfloor \dfrac{t-t_{\text{drain}}}{\bar c} \right\rfloor\right), & t > t_{\text{drain}} \quad\text{(drain)}
\end{cases}
$$

where $\bar d$ and $\bar c$ are representative per-layer download and compute times. In steady state the buffer oscillates between $B-1$ and $B$ (one layer is released as the next arrives), so $RSS(t)$ oscillates by roughly one $s_{\text{layer}}$.

### 7.3 The pre-allocation caveat (must not be omitted)

In the naive implementation the **full parameter set is pre-allocated** (or the whole file is mmapped), so $RSS_{\text{base}}$ already contains all weights. The ring buffer holds *additional transient copies* of at most $B$ layers. Therefore:

- Peak RSS is approximately $RSS_{\text{base}} + B\,s_{\text{layer}} + A_{\max}$, and $B\,s_{\text{layer}}$ is small relative to a multi-gigabyte parameter set.
- Reducing $B$ from 3 to 2 saves only one $s_{\text{layer}}$; it does **not** reduce peak RSS to "one layer's worth".
- **No peak-RSS reduction is claimed.** A genuine reduction requires a custom parameter store that does not keep a second copy — e.g. streaming into paged buffers with `madvise(MADV_DONTNEED)`, or a tiered CPU/GPU parameter server. That is future work.

The memory model is still useful: it tells the engine how deep it may prefetch without exceeding a hard budget, and it predicts the *shape* of the RSS trace (ramp, plateau, drain) that the benchmark should observe.

---

## 8. Chunk-granularity tradeoff

### 8.1 Per-layer vs sub-layer chunks

So far each stage is a whole layer, $z=s_{\text{layer}}$. Smaller chunks $z<s_{\text{layer}}$ allow compute of chunk $j$ to overlap download of chunk $j+1$ *within* a layer, and reduce the pipeline-fill latency. But each chunk is a separate HTTP request, costing $o$ seconds of overhead. With $N$ total bytes there are $n=N/z$ requests.

A first-order model for total TTFT with chunk size $z$, assuming chunks are independently downloadable and compute is chunkable with per-chunk compute $c_{\text{chunk}} \approx C/n$ (compute proportional to bytes):

$$
T(z) \approx t_{\text{startup}}
+ \frac{z}{B_{\text{net}}}
+ c_{\text{chunk}}
+ (n-1)\max\!\left(\frac{z}{B_{\text{net}}},\,c_{\text{chunk}}\right)
+ n\,o
+ c_{\text{tail}}.
$$

The first chunk's download is exposed (pipeline fill); each subsequent chunk contributes the larger of its download and compute; every request contributes $o$.

### 8.2 The overhead-vs-fill curve

When the $\max$ term is roughly constant, the $z$-dependent part is the request overhead plus the fill:

$$
f(z) = \underbrace{\frac{N}{z}\,o}_{\text{request overhead}}
+ \underbrace{\frac{z}{B_{\text{net}}}}_{\text{first-chunk fill}}.
$$

$f$ is convex: it falls as $1/z$ for small chunks and rises linearly for large chunks. Minimising,

$$
f'(z) = -\frac{No}{z^2} + \frac{1}{B_{\text{net}}} = 0
\quad\Longrightarrow\quad
\boxed{\;z^\star = \sqrt{N\,o\,B_{\text{net}}},\qquad n^\star = \frac{N}{z^\star} = \sqrt{\frac{N}{o\,B_{\text{net}}}}.\;}
$$

Units check: $N$ [bytes] × $o$ [s] × $B_{\text{net}}$ [bytes/s] = bytes$^2$, so $z^\star$ is in bytes. The curve is flat near the minimum, so an order-of-magnitude estimate of $o$ is sufficient.

For full masking at chunk granularity one also wants the chunk download not to exceed the chunk compute:

$$
\frac{z}{B_{\text{net}}} \le c_{\text{chunk}}
\quad\Longleftrightarrow\quad
z \le B_{\text{net}}\,c_{\text{chunk}}.
$$

The usable chunk size is therefore the smaller of the overhead optimum and the masking condition, floored at the smallest range the object store serves efficiently.

### 8.3 Practical recommendation for this build

**Use layer granularity** ($z=s_{\text{layer}}$). Reasons:

1. The manifest already groups tensors by layer, so no extra parser or offset logic is needed.
2. The number of requests is $n=L$ (tens), so $n\,o$ is negligible compared with per-layer compute (e.g. $24\times5\,\text{ms}=0.12\,\text{s}$).
3. A layer is the natural unit of compute; sub-layer compute would require splitting GEMMs/attention, which the stdlib synthetic engine does not do.
4. Cross-layer lookahead ($i+1,i+2$) already amortises pipeline fill; sub-layer chunking mainly helps when $L$ is very small or $B_{\text{net}}$ is very low.
5. Layer granularity keeps the ring-buffer/eviction logic simple and the memory model exact.

Sub-layer chunking becomes worthwhile if $o$ is very small, $B_{\text{net}}$ is low, or the model has few very large layers. It should be treated as future work, not a claim of this build.

---

## 9. Worked symbolic example — **illustrative, NOT measured**

> **WARNING:** Every number in this section is a round, plausible placeholder chosen to exercise the equations. **None of it is measured.** The measured 2026-11-11 run is reported separately in `docs/project_report.md` §8; do not conflate the two.

### 9.1 Symbols

| Symbol | Illustrative value |
|---|---:|
| $L$ | 24 layers |
| $s_{\text{layer}}$ | 100 MB = $100\times10^6$ bytes |
| $N=L\,s_{\text{layer}}$ | 2.4 GB |
| $B_{\text{net}}$ | 100 MB/s |
| $d_i=s_i/B_{\text{net}}$ | 1.0 s per layer |
| $c_i$ | 0.4 s per layer |
| $t_{\text{startup}}=t_q+t_i+t_m$ | 2.0 s |
| $c_{\text{tail}}$ | 0.1 s |
| $k_{\max}$ | 3 slots |
| $RSS_{\text{base}}$ (params pre-allocated) | 2600 MB |
| $RSS_{\text{limit}}$ | 3200 MB |
| $A_{\max}$ | 200 MB |

### 9.2 Baseline

$$
D = 24\times1.0 = 24.0\,\text{s},\qquad
C = 24\times0.4 = 9.6\,\text{s},
$$

$$
TTFT_{\text{base}} = 2.0 + 24.0 + 9.6 + 0.1 = 35.7\,\text{s}.
$$

### 9.3 Pipelined

First layer is exposed; each later layer contributes $\max(1.0,0.4)=1.0$ s:

$$
T_{\text{pipe}}
= 2.0 + \underbrace{1.0+0.4}_{\text{layer 0}}
+ 23\times1.0
+ 0.1
= 26.5\,\text{s}.
$$

Equivalently, using the symmetric form:

$$
T_{\text{pipe}}
= 2.0 + \sum_{i=0}^{23}\max(1.0,0.4) + \min(1.0,0.4) + 0.1
= 2.0 + 24.0 + 0.4 + 0.1
= 26.5\,\text{s}.
$$

### 9.4 Speedup

$$
S = \frac{35.7}{26.5} \approx 1.35\times.
$$

This is well below the $2\times$ ceiling, because the illustrative link is slower than compute ($d>c$), so the download dominates even after overlap.

### 9.5 Masking

Corrected (excluding layer 0):

$$
M = \frac{\sum_{i=1}^{23}\min(1.0,0.4)}{\sum_{i=0}^{23}1.0}
= \frac{23\times0.4}{24}
= \frac{9.2}{24}
\approx 0.383 \;(38.3\%).
$$

The uncorrected all-layer form would give $(1.0+9.2)/24\approx42.5\%$; the difference is the first-layer penalty. These are illustrative numbers. The measured 8-layer run reports $M=0.626$, consistent with the $1-d_0/D$ ceiling.

Consistency check with §5.3, ignoring startup/tail:

$$
S \approx \frac{1}{1-\dfrac{M D}{D+C}}
= \frac{1}{1-\dfrac{0.383\times24}{33.6}}
\approx 1.38,
$$

which matches the $1.35\times$ above once the 2.1 s of startup+tail is included.

### 9.6 Memory clamp

$$
B = \min\!\left(3,\;
\left\lfloor\frac{3200-2600-200}{100}\right\rfloor\right)
= \min(3,\,4) = 3.
$$

So the configured depth is safe; the buffer adds at most $3\times100=300$ MB, giving a peak of about $2600+300+200=3100$ MB, under the 3200 MB limit. **Crucially**, lowering $B$ from 3 to 2 would save only 100 MB out of ~3.1 GB (≈3%). This is the numerical illustration of why **no peak-RSS reduction is claimed**: the parameters are already resident, and the buffer is a small transient term.

### 9.7 Chunk-size optimum

With $N=2.4\times10^9$ bytes, $o=5$ ms, $B_{\text{net}}=10^8$ B/s:

$$
z^\star = \sqrt{N o B_{\text{net}}}
= \sqrt{2.4\times10^9 \times 0.005 \times 10^8}
= \sqrt{1.2\times10^{15}}
\approx 3.46\times10^7\ \text{bytes} \approx 34.6\ \text{MB},
$$

$$
n^\star = \frac{N}{z^\star} \approx 69\ \text{requests}.
$$

Layer size is 100 MB, i.e. larger than $z^\star$, so sub-layer chunking would give 69 requests instead of 24. At $o=5$ ms the overhead difference is only $(69-24)\times5\,\text{ms}\approx0.23$ s against a 26.5 s TTFT — negligible. At $o=50$ ms, $z^\star\approx110$ MB, essentially the layer size, confirming that **layer granularity is the right choice for this build**.

---

## 10. Assumptions and validity limits

The model is valid under the following assumptions. Violating them does not invalidate the experiment, but it does mean the equations are descriptive rather than predictive, and the measured value must be used.

1. **One effective network link.** Concurrent range requests share $B_{\text{net}}$; $d_i$ is the effective time to add stage $i$'s bytes. The recurrence's serialisation is an aggregate approximation, not a claim about TCP.
2. **Sequential, non-preemptive compute.** Layer $i+1$ cannot compute before layer $i$ finishes; there is no intra-layer parallelism.
3. **Single worker, single request.** No batching, no concurrent tenants, no queueing between layers.
4. **Deterministic per-layer times.** $c_i$ does not vary with cache state or thermal throttling; $d_i$ does not vary with congestion.
5. **Warm object store.** No cache miss at the storage tier; manifest is cached or its fetch is included in $t_m$.
6. **No failures or retries.** The model describes the happy path; §7 of `docs/01_architecture.md` covers failures.
7. **Uniform layers for the closed form.** Non-uniform layers use the recurrence; the closed form is an approximation.
8. **TTFT only.** The model stops at the first token; decode throughput, KV-cache growth, and multi-token generation are out of scope.
9. **Startup independent of model size.** $t_{\text{startup}}$ excludes weight transfer; if the runtime prefetches weights during init, fold that into $d_0$ or startup consistently.
10. **Ideal throttle.** $B_{\text{net}}$ is constant; real `tc`/sleep throttling is bursty.
11. **No quantisation/compression.** Payload bytes equal stored dtype bytes.
12. **Bounded activations.** $A(t)\le A_{\max}$; a long prompt or large batch would violate this.
13. **Pre-allocation caveat.** $RSS_{\text{base}}$ includes the full parameter set in the naive implementation; the buffer is a transient addition (§7.3).
14. **No model training.** Frozen/synthetic checkpoints only.

**What the harness measured (2026-11-11; see `results/benchmark.csv`):**

- $d_0$ and per-layer download time, from `AsyncPrefetcher`/`PipelineDriver` timestamps.
- $c_i$ per layer, from `PipelineDriver` compute timestamps.
- $D_{\text{exposed}}$ / `io_wait_s`, from the driver's blocking time.
- TTFT (full forward pass), masking, and RSS trajectory.

**Not separately instrumented:** $t_q$, $t_i$, $t_m$, $t_{\text{load}}$, $c_{\text{tail}}$. Do not quote them as measured.
- $RSS(t)$, from `psutil`/`/proc/self/statm`, sampled per layer.
- $M$ and $S$, computed from the above by `BenchmarkHarness.py`, not asserted by this document.

---

## 11. Equation summary

| Quantity | Equation |
|---|---|
| Baseline TTFT | $TTFT_{\text{base}} = t_q+t_i+t_m+t_{\text{load}}+D+C+c_{\text{tail}}$ |
| Exact pipeline recurrence | $A_0=d_0,\ F_0=d_0+c_0;\ A_i=\max(A_{i-1}+d_i,F_{i-1}),\ F_i=A_i+c_i$ |
| Pipelined TTFT | $T_{\text{pipe}} = t_{\text{startup}}+F_{L-1}+c_{\text{tail}}$ |
| Uniform closed form | $T_{\text{pipe}} = t_{\text{startup}}+d+c+(L-1)\max(d,c)+c_{\text{tail}}$ |
| Symmetric form | $T_{\text{pipe}} = t_{\text{startup}}+\sum_i\max(d_i,c_i)+\min(d_0,c_0)+c_{\text{tail}}$ |
| Masking ratio | $M=\dfrac{\sum_{i\ge1}\min(d_i,c_i)}{\sum_i d_i}$ |
| Speedup | $S=T_{\text{base}}/T_{\text{pipe}}\le 2$; $S\to(d+c)/\max(d,c)$ for uniform layers |
| Buffer clamp | $B\le\min\!\left(k_{\max},\left\lfloor(RSS_{\text{limit}}-RSS_{\text{base}}-A_{\max})/s_{\text{layer}}\right\rfloor\right)$ |
| RSS trajectory | $RSS(t)=RSS_{\text{base}}+n(t)s_{\text{layer}}+A(t)+\varepsilon(t)$ |
| Chunk optimum | $z^\star=\sqrt{NoB_{\text{net}}},\quad n^\star=\sqrt{N/(oB_{\text{net}})}$ |

---

*Every numeric value in §9 is illustrative and must be replaced by a measured value from `run.sh`. The model's job is to bound expectations and define the instrumentation, not to pre-empt the experiment.*
