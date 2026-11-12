# Async Progressive Layer-Streaming Engine

**IS355TBD (Cloud Computing), 5th Sem, RVCE — Experiential Learning project**

Design and empirical evaluation of an asynchronous progressive layer-streaming engine for
accelerating serverless LLM invocations. The engine treats a safetensors checkpoint as
per-layer byte ranges, fetches layer `i+1` over HTTP Range while computing layer `i`, and
bounds the transient working set with a ring buffer and RSS clamp.

## Quick start

```bash
python3 -m pip install aiohttp psutil requests   # one package install command
bash setup.sh                                    # idempotent; generates synthetic models
bash run.sh                                      # 3 baseline + 3 pipeline runs -> results/
```

Minimal proof of the streaming pipeline (stdlib only):

```bash
python3 examples/minimal_stream.py
```

## Measured result (2026-11-11)

Synthetic 8-layer F32 checkpoint (8.00 MiB), local RangeServer over HTTP loopback,
8 Mbit/s shared throttle, 3 runs per mode:

| Metric (mean) | Monolithic | Pipelined | Change |
|---|---:|---:|---:|
| TTFT, full forward (s) | 15.702 | 10.816 | **−31.1% (1.45×)** |
| First-layer fill (s) | 9.582 | 3.959 | diagnostic only, not TTFT |
| I/O masking ratio | 0.000 | 0.626 | — |
| Peak RSS (MB) | 53.67 | 48.28 | reported, **not claimed as a memory win** |

Source: `results/benchmark.csv`. No number is fabricated. TTFT means completion of the
full forward pass; a real first token requires every layer.

## Documentation

| File | Contents |
|---|---|
| `docs/PROJECT_SUBMISSION.md` | Consolidated submission-ready document |
| `docs/project_report.md` | Full EL report (11 sections, literature, results) |
| `docs/01_architecture.md` | Architecture and safetensors random-access design |
| `docs/02_math_model.md` | TTFT, masking, speedup, memory and chunking model |
| `docs/viva_qa.md` | Viva Q&A with measured answers |
| `docs/build_todo.md` | Build status and verification evidence |

## Honest limitations

- Compute is a deterministic synthetic FP32 kernel, **not a transformer**.
- Local RangeServer + loopback is a stand-in for **MinIO/S3 + `tc`**; it is not a real
  serverless deployment and does not measure cold-start provisioning.
- **No peak-RSS saving is claimed**: the naive implementation pre-allocates the full
  parameter set; the measured RSS gap reflects whole-file retention vs a bounded buffer.
- Masking is **62.6%**, below 70%, and specific to this synthetic workload and link.
- `torch`, `numpy`, and `safetensors` are not required and are not installed.

## Layout

```
src/core/       safetensors parser, manifest writer, prefetcher, buffer, pipeline driver
src/storage/    stdlib HTTP Range server with bandwidth throttle
src/bench/      benchmark harness + synthetic model generator
examples/       minimal end-to-end streaming demo
docs/           architecture, math, report, viva, consolidated submission
results/        benchmark.csv, per_layer.csv, RSS traces
run.sh          start server, run benchmark, clean up
setup.sh        idempotent deps + synthetic models
scripts/        verify_build.sh merge-review verifier
```
