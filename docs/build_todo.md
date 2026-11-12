# Build TODO — Async Progressive Layer-Streaming Engine

**Project:** Design and Empirical Evaluation of an Asynchronous Progressive Layer-Streaming Engine for Accelerating Serverless Large Language Model Invocations
**Course:** IS355TBD (Cloud Computing), 5th Sem, RVCE — Experiential Learning
**Coordinator:** `lead` agent
**Last updated:** 2026-11-11 (post-run verification)

## Environment snapshot (verified)

| Item | Value |
|---|---|
| Python | 3.14.9 (one subagent saw 3.14.8 in its isolated process; both work) |
| aiohttp | 3.14.5 — installed by `setup.sh` |
| psutil | 7.2.2 — installed |
| requests | 2.34.2 — installed |
| numpy / torch / safetensors | **MISSING — not required** |
| curl | available |

Design consequence: the streaming core runs on the **Python standard library only**
(`struct`, `mmap`, `array`, `asyncio`, `http.server`, `urllib`). `torch`/`numpy` are not
required. Synthetic layer compute stands in for transformer GEMMs and is disclosed.

## Workstream status

| WS | Owner | Deliverables | Status | Evidence |
|---|---|---|---|---|
| WS-1 | architect | `docs/01_architecture.md`, `docs/02_math_model.md` | **COMPLETE** | 403 + 583 lines; SAFETENSORS layout, pipeline math, masking bound, RSS clamp, chunk optimum. Notation reconciled (`S`=speedup, `T_startup`, `N_look`). |
| WS-2 | engineer-core | `src/core/*.py`, `src/core/README.md`, `examples/minimal_stream.py` | **COMPLETE** | `py_compile` OK; minimal example exited 0: full-forward TTFT 76.88 ms, first-layer 36.24 ms, 5 layers, 49.6% masking on the tiny demo. Stdlib-only. |
| WS-3 | engineer-bench | `src/storage/*`, `src/bench/*`, `run.sh`, `setup.sh`, `requirements.txt` | **COMPLETE** | Range 206/416/HEAD/suffix tests pass; throttle verified; `run.sh` exit 0; 3 baseline + 3 pipeline rows in `results/benchmark.csv`. |
| WS-4 | writer | `docs/project_report.md`, `docs/viva_qa.md` | **COMPLETE** | 550 + 202 lines; five papers cited correctly; measured results filled in by lead from CSV; no fabricated numbers. |
| WS-5 | lead | `docs/build_todo.md`, `docs/PROJECT_SUBMISSION.md`, merge review | **COMPLETE** | TTFT definition corrected to full forward pass; harness re-run; consolidated submission written; verifier passes. |

## Measured results (authoritative: `results/benchmark.csv`)

Run 2026-11-11, synthetic 8-layer F32 checkpoint (8.00 MiB), 8 Mbit/s shared throttle,
lookahead 2, work-factor 1.0, 3 runs/mode in fresh subprocesses.

| Metric (mean) | Monolithic | Pipelined | Change |
|---|---:|---:|---:|
| TTFT (full forward, s) | 15.702 | 10.816 | **−31.1% (1.45×)** |
| First-layer fill (s) | 9.582 | 3.959 | −58.7% (diagnostic only) |
| I/O masking ratio | 0.000 | 0.626 | — |
| Peak RSS (MB) | 53.67 | 48.28 | reported, **not claimed as a memory win** |
| Bytes downloaded | 8,389,896 | 8,388,608 | header is excluded from layer ranges |

**TTFT definition:** time to complete the full forward pass (a real first token needs every
layer). `first_layer_s` is streaming-fill latency and is **not** an LLM TTFT.

## Merge-review checklist (closed)

- [x] Every promised file exists at the exact path
- [x] `python3 -m py_compile` passes on every `.py` artifact
- [x] `python3 examples/minimal_stream.py` runs end-to-end (stdlib only)
- [x] `bash setup.sh` idempotent; `bash run.sh` writes `results/benchmark.csv`
- [x] No fabricated benchmark numbers; measured values trace to the CSV
- [x] Every engineer file header contains a `STATUS:` line
- [x] Docs state the local RangeServer + loopback substitution for MinIO + `tc`
- [x] Known limitations consistent across docs
- [x] TTFT mislabel fixed (first-layer → full-forward); benchmark re-run

## Honesty guardrails (binding)

1. No invented benchmark numbers. Unmeasured = "not instrumented" or "TBD".
2. Do **not** claim peak-RSS savings: the naive implementation pre-allocates full params;
   the measured 53.7 → 48.3 MB gap reflects the synthetic baseline holding the whole file.
   The defensible wins are TTFT and I/O masking.
3. Measured masking is **62.6%**, below 70%, and is specific to this synthetic workload/link.
4. No full model was trained; the synthetic compute is a real MAC loop but not a transformer.
5. Local RangeServer + loopback is not a serverless deployment; production path is MinIO + `tc`.
