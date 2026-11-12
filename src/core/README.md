# `src/core` — Async Progressive Layer-Streaming Engine

A working, standard-library-first engine that streams a `safetensors`
checkpoint one layer at a time over HTTP Range while computing the previous
layer. It is the artifact for the IS355TBD Cloud Computing EL project.

STATUS: Built: five core modules + this README + runnable minimal example.
TBD: MinIO/boto3 upload untested; real-transformer GEMMs require torch/llama.cpp (not installed). Measured benchmark results are in `results/benchmark.csv`.
Packages: stdlib only; aiohttp/psutil/boto3 optional; torch/numpy NOT required.

> **Honesty note.** The compute engine is a deterministic synthetic
> matvec/elementwise kernel, not a transformer. The artifact demonstrates the
> I/O/compute pipeline, TTFT and I/O masking on CPU. It does **not** claim
> transformer GEMM speed, full-model training, or peak-RSS savings from
> pre-allocating all parameters.

## Pipeline

```
S3/MinIO (or local RangeServer)
        │  HTTP Range: bytes=start-end
        ▼
AsyncPrefetcher ──► MemoryBuffer (bounded FIFO, RSS clamp)
        │                       │
        │  lookahead N+1        │ get / evict
        ▼                       ▼
PipelineDriver ──► SyntheticComputeEngine (FP32 matvec + elementwise)
        │
        ▼
metrics: ttft_s, per_layer, io_wait_s, compute_s, masking_ratio, rss_trace
```

## Files

| File | Responsibility |
|---|---|
| `SafetensorsManifest.py` | Parse the 8-byte header length + JSON table with `mmap`; expose `TensorInfo` absolute byte ranges and `LayerRange` groups. |
| `S3ManifestWriter.py` | Write a JSON chunk manifest (`layer -> key/start/end/length/tensors`) and optionally upload it with boto3. |
| `AsyncPrefetcher.py` | asyncio producer with an N+1 lookahead window, HTTP Range fetch, 206/Content-Range/length verification, retries/backoff, adaptive pacing, stats. |
| `MemoryBuffer.py` | Bounded FIFO payload buffer with explicit eviction/drop policy and RSS-based clamping. |
| `PipelineDriver.py` | Consume layer *i*, compute, evict, record timings/RSS; returns the metrics dict. |
| `__init__.py` | Public exports. |

There is deliberately **one** module README instead of five duplicated ones.

## Safetensors random access

The first 8 bytes are a little-endian `uint64` header length `N`. The next `N`
bytes are UTF-8 JSON (space-padded to an 8-byte boundary). Each non-metadata
entry looks like:

```json
"model.layers.0.self_attn.q_proj.weight": {
  "dtype": "F32", "shape": [4096, 4096], "data_offsets": [0, 67108864]
}
```

`data_offsets` are relative to the data section at `8 + N`. The parser adds
that base to produce `abs_start`/`abs_end`, which are exactly what an HTTP
`Range: bytes=abs_start-(abs_end-1)` request needs.

Layer grouping uses the top-level module prefix: `model.layers.0`,
`model.embed_tokens`, `model.norm`, `lm_head`. Layers are sorted naturally
(`layers.2` before `layers.10`). `__metadata__` is skipped as a tensor.

## Quick start

```python
import asyncio
from src.core import (
    AsyncPrefetcher, MemoryBuffer, PipelineDriver,
    S3ManifestWriter, SafetensorsManifest, SyntheticComputeEngine,
)

manifest = SafetensorsManifest("models/tiny.safetensors")
S3ManifestWriter(manifest, "models/tiny.safetensors").write("chunk_manifest.json")

async def main():
    buffer = MemoryBuffer(max_depth=3, rss_limit_mb=1024)
    prefetcher = AsyncPrefetcher(
        "http://127.0.0.1:8000/tiny.safetensors",
        manifest,
        buffer=buffer,
        lookahead=2,
    )
    driver = PipelineDriver(
        prefetcher, SyntheticComputeEngine(work_factor=2.0), buffer=buffer
    )
    async with prefetcher:
        metrics = await driver.run()
    print(metrics["ttft_s"], metrics["masking_ratio"])

asyncio.run(main())
```

Run the mandatory end-to-end demo from the repository root:

```bash
python3 examples/minimal_stream.py
```

It builds a tiny 2-layer FP32 safetensors file, serves it from a threaded
stdlib HTTP server with Range support, writes the chunk manifest, streams every
layer, and prints TTFT, per-layer download/compute/RSS, and summary stats.

## Interface contract

```python
SafetensorsManifest(path_or_bytes)
    .tensors: dict[str, TensorInfo]   # abs_start/abs_end absolute offsets
    .layers: dict[str, LayerRange]    # .tensors, .start, .end, .length

S3ManifestWriter(manifest, object_key).write(path) -> dict

AsyncPrefetcher(url, layers_or_manifest, buffer=None, lookahead=2, ...)
    await start(); await get(name) -> bytes; await stop(); .stats()
    # get_payload(name) -> LayerPayload (data + timings) is the richer variant

MemoryBuffer(max_depth=..., rss_limit_mb=...)
    put(layer, payload) -> bool; get(layer); evict(layer); stats()

PipelineDriver(prefetcher, compute_engine=None, buffer=None)
    await run(layer_names=None) -> metrics dict
```

### Semantics and policies

* **Ranges.** `start` is inclusive, `end` exclusive. A layer request is
  `bytes=start-(end-1)` and the body must be exactly `end - start` bytes.
* **Lookahead.** The producer window is `lookahead + 1` layers. It launches up
  to that many independent Range requests at once, so downloads continue while
  the driver computes. When compute-bound the window shrinks toward 2 (enough
  to overlap the next fetch with the current forward pass); when download-bound
  it stays at the full window.
* **Compute scheduling.** `PipelineDriver` runs the CPU-bound synthetic kernel
  via `asyncio.to_thread` by default, keeping the event loop free to schedule
  prefetches. Set `compute_in_thread=False` only for a custom non-thread-safe
  engine.
* **Buffer overflow.** Evict the oldest *unreferenced* entry. If every older
  entry is pinned, drop the incoming payload. A dropped prefetch is not fatal:
  `get()` re-fetches on demand. There is no blocking `put()`, so a slow consumer
  cannot deadlock the producer.
* **RSS clamp.** When process RSS exceeds `rss_limit_mb`, evict until the limit
  is met or no eviction is possible. This is best-effort: CPython may retain
  freed memory in its allocator, so RSS can remain above the limit.
* **Concurrency.** `MemoryBuffer` is asyncio-only and not thread-safe; its
  methods are synchronous and contain no `await`, so they are safe on the
  single event-loop thread. `AsyncPrefetcher` and `PipelineDriver` are async.
* **TTFT.** Time from `run()` start to completion of the **full forward pass**
  (the last layer's compute), because a real first token needs every layer.
  `first_layer_s` records layer 0's compute completion and is streaming-fill
  latency, not an LLM TTFT; `first_payload_s` is also reported. See
  `PipelineDriver.py`.
* **Masking.** `masking_ratio = compute_s / (compute_s + io_wait_s)` in [0, 1];
  `io_masking_ratio = 1 - io_wait_s / total_s` is also reported. Measured
  values are reported, never assumed.
* **Metrics shape.** `metrics["layers"]` is the number of consumed layer groups;
  `metrics["layer_names"]` is the ordered list and `metrics["per_layer"]` holds
  per-layer rows. `bytes_downloaded` is the sum of consumed payload sizes;
  `metrics["prefetcher_stats"]["bytes_downloaded"]` is the network total
  (including any re-fetches).

## Packages

* **Required: none beyond the Python 3.14 standard library.**
* Optional: `aiohttp` (faster HTTP backend; `urllib` fallback is used when it is
  absent), `psutil` (precise current RSS; `/proc/self/statm` and `getrusage`
  fallbacks exist), `boto3` (only for `S3ManifestWriter.upload`).
* **Not required:** `torch`, `numpy`, `safetensors`.

The verification run used `aiohttp` 3.14.5 and `psutil` 7.0.0 when they became
available; the `urllib` fallback was also exercised while `aiohttp` was briefly
absent from the interpreter. The example therefore works with or without
`aiohttp`.

## Known limitations

1. Synthetic compute is not a transformer; absolute latency numbers are not
   transformer latency.
2. The naive implementation still pre-allocates full model parameters for a
   real model, so peak-RSS savings are not claimed. The defensible wins are
   TTFT and I/O masking; a custom parameter store is future work.
3. A layer span may contain multiple dtypes; the synthetic kernel uses the
   layer's shared dtype when known, else FP32, and falls back to byte-wise
   arithmetic for unsupported dtypes.
4. The local HTTP RangeServer over loopback is a stand-in for MinIO + `tc`;
   production would use object storage, TLS/auth, and bandwidth shaping.
5. One HTTP request per layer; sub-layer chunk splitting and HTTP/2
   multiplexing are not implemented.
6. The buffer is not thread-safe and `S3ManifestWriter.upload` is not exercised
   without a MinIO/S3 endpoint.

## STATUS

* Built: all five modules, one README, and `examples/minimal_stream.py`.
* TBD: real-transformer benchmark numbers require torch/llama.cpp (not
  installed); the synthetic benchmark is in `results/benchmark.csv`. MinIO/boto3
  upload is untested locally.
* Packages: stdlib only; aiohttp/psutil/boto3 optional; torch/numpy NOT
  required.
