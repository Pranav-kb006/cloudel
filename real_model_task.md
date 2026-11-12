# real-model benchmark task (for OpenCode agents)

Project dir: /home/bhandi/opencode-test

Goal: swap in the REAL model (TinyLlama-1.1B-Chat-v1.0, models/tinyllama.safetensors,
2.098 GB safetensors, 22 Transformer layers, hidden 2048, head_dim 64, kv_heads=4) and
produce a COMPLETE end-to-end benchmark result file.

Deliverables:
  1. results/real_model_end_to_end.csv  — real model per-layer + summary rows.
  2. A short results summary (either a 1-pager in results/README.md or in this task),
     with:
     - real per-layer compute_s (seq=1 forward pass, CPU torch 2.15.0+cpu),
     - first-layer TTFT (s) for the progressive vs baseline run,
     - masking_ratio (real engine),
     - a clear note that the runtime uses CPU (torch.cuda.is_available()=False);
       the engine will use CUDA if device='cuda' is passed and CUDA is available.

Machine: RTX 4050, 6 GB VRAM, CUDA 13.0 available, but torch CPU wheel is installed
(torch.cuda.is_available() is False). Do NOT claim CUDA is active; state the runtime
truth.

What is already done (do NOT redo): 
  - RealModelComputeEngine.py written and unit-verified: 22-layer real forward pass
    completes (checksums finite and rising 0.024 -> 0.587), ~0.09-0.17 s/layer.
  - models/tinyllama.safetensors downloaded cleanly (~2.098 GB).
  - torch 2.15.0+cpu installed in the venv.

The remaining bugs in real_benchmark.py (fix, do NOT rewrite from scratch):
  1. In main(), `dl` is referenced in the CSV write line but never defined (the
     baseline `run_baseline` function was deleted but its body is still there).
  2. The baseline block mixed calls to a deleted `run_baseline` and references to
     variables (`masking`, etc.) that are not in scope.

The FallbackPrefetcher/FallbackDriver per-layer keys (from BenchmarkHarness.py lines
98-110 and 398-408) are:
  ["layer","bytes","fetch_s","compute_s","blocked_s","fetch_start_s","fetch_end_s",
   "compute_start_s","compute_end_s"]

Use these keys for the per-layer CSV rows. For the summary rows, read from the driver
result dict `real` (keys: first_layer_s, io_wait_s, compute_s, total_s, masking_ratio).

Important: the pipeline RUN already works (it ran 22 rows, compute total ~2.3s).
You only need to fix the script so it runs CLEANLY end-to-end and writes the CSV.

Steps:
  1. Read real_benchmark.py.
  2. Read BenchmarkHarness.py to confirm per-layer + summary keys (done above).
  3. Rewrite real_benchmark.py cleanly: progressive run with REAL engine, then a
     baseline (full pre-download, then run REAL.engine on all 22 layers), write the
     CSV with per-layer rows + summary rows.
  4. Run it: venv/bin/python3 real_benchmark.py
  5. Confirm it exits 0 and produces results/real_model_end_to_end.csv.
  6. Print the real numbers: per-layer compute_s, first-layer TTFT, masking_ratio,
     io_wait, total, and a one-line note that runtime is CPU (CUDA not active).

Run command to use for the benchmark:
  /home/bhandi/.hermes/installs/6dbeb051a43d0149/environments/f5e1f161ceda4ec89c8cf1e30609f5db/venv/bin/python3 real_benchmark.py
