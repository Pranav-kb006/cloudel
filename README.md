# Progressive AI — Asynchronous Layer-Streaming Engine for Serverless LLMs

An asynchronous progressive layer-streaming engine that accelerates serverless Large Language Model (LLM) cold starts by disaggregating SafeTensors checkpoints into per-layer HTTP byte ranges, overlapping network prefetching with PyTorch Transformer execution, and evicting completed layers to bound container working-set memory.

## Architecture

- **Header-Only Remote Manifest (`src/storage/s3_adapter.py`)**: Resolves per-layer byte offsets using two small RFC 7233 HTTP `Range` requests (`bytes=0-7` and `bytes=8-(8+H-1)`) without downloading the full checkpoint.
- **Asynchronous Producer-Consumer Pipeline (`src/bench/ablation_suite.py`, `src/core/`)**: Semaphore-bounded (`K`) HTTP Range prefetcher streaming layer `i+1` concurrently while executing layer `i`, with immediate post-layer payload eviction.
- **PyTorch Transformer Compute Engine (`src/core/RealModelComputeEngine.py`)**: Zero-copy `torch.frombuffer` BF16/FP16/FP32 weight binding executing RMSNorm, RoPE, Causal Grouped-Query Attention (GQA), and SwiGLU MLP.
- **Cloud-Native Deployment (`Dockerfile`, `docker-compose.yml`)**: Containerized worker with cgroup memory/CPU limits paired with S3-compatible MinIO object storage and a shared token-bucket HTTP `RangeServer`.
- **Observability Dashboard (`web/`)**: Light-first cloud infrastructure dashboard with live side-by-side Monolithic vs. Progressive execution, Model Registry, Experiment History (CSV export), Latency/Memory/Network Analytics, Execution Pipeline view, and System Logs.

## Quick Start

```bash
pip install -r requirements.txt
pip install torch numpy matplotlib

# 1. Generate the 12-layer Llama BF16 safetensors checkpoint (290 MiB, 152M params)
python src/bench/make_transformer_model.py

# 2. Launch the observability & live experiment dashboard (http://127.0.0.1:8000)
python web/demo_server.py
```

### Run Empirical Benchmarks & Generate Charts

```bash
python src/bench/ablation_suite.py
python src/bench/plot_charts.py
```

### Run with Docker & MinIO S3

```bash
docker compose up --build
```

## Measured Benchmarks (12-Layer Llama BF16, 290 MiB, `seq_len=128`, `K=2`)

| Link Bandwidth | Monolithic TTFT | Progressive TTFT | TTFT Gain | 1st-Layer Speedup | I/O Masking | Working-Set ΔRSS Saved |
|---:|---:|---:|---:|---:|---:|---:|
| 100 Mbps | 27.60 s | 23.89 s | −13.4% (1.16×) | 12.9× | 4.1% | −59.7% (400.4 → 161.3 MB) |
| 250 Mbps | 12.14 s | 9.89 s | −18.5% (1.23×) | 12.0× | 18.8% | −61.9% (341.9 → 130.3 MB) |
| 500 Mbps | 8.25 s | 5.49 s | **−33.5% (1.50×)** | **13.8×** | 23.0% | −39.1% (341.1 → 207.6 MB) |
| 1000 Mbps | 3.83 s | 3.37 s | −12.0% (1.14×) | 9.6× | **54.6%** | −56.4% (368.5 → 160.6 MB) |

## Repository Structure

```text
src/core/       PyTorch Llama compute engine, safetensors manifest, prefetcher, ring buffer
src/storage/    RFC 7233 throttled RangeServer, header-only S3/MinIO adapter
src/bench/      Transformer checkpoint generator, ablation suite, chart plotting
web/            Observability dashboard backend (demo_server.py) and light-first frontend
results/        Measured CSV/JSON benchmark outputs and 300-DPI charts
Dockerfile      Containerized serverless worker image
docker-compose.yml  MinIO S3 + streaming worker stack
```
