# BRIEF: Build "Async Progressive Layer-Streaming Engine" project — SIMULTANEOUS PARALLEL BUILD

## Project
Title: "Design and Empirical Evaluation of an Asynchronous Progressive Layer-Streaming Engine for Accelerating Serverless Large Language Model Invocations"
Course: IS355TBD (Cloud Computing), 5th Sem, RVCE — Experiential Learning (EL) project.
Goal: Don't document it later — BUILD the working artifact(s) now. Submission is tomorrow.

## Your role
You are the `lead` agent — project coordinator. Split the work into PARALLEL workstreams, dispatch specialist agents to work simultaneously, collect their output, and synthesize the final deliverable.

## Parallel workstreams (all run at the same time, independent agents)

### WS-1 — architect (records/architecture/model)
- Write the architecture doc: the 3-stage async pipeline (Client → Serverless Worker + Pipelined Engine + S3/MinIO storage), with the ring buffer, tokenizer/embeddings, compute engine, prefetcher daemon.
- Write the SAFETENSORS random-access explanation: 8-byte header length + JSON metadata + byte-offset tensor table.
- Write the math model: baseline latency equation + pipelined TTFT equation + per-layer overlap equation. Use clean LaTeX. Keep it focused on TTFT, I/O masking ratio, memory trajectory, and chunk-granularity tradeoff.
- Produce: `docs/01_architecture.md`, `docs/02_math_model.md`

### WS-2 — engineer-core (streaming engine code)
- Build in `src/core/`: safetensors parser (`SafetensorsManifest.py` → parse header+metadata, expose per-tensor byte ranges), chunk manifest writer (`S3ManifestWriter.py` → writes a manifest mapping layer name → byte range + S3 key), async prefetcher (`AsyncPrefetcher.py` → asyncio queue, ring buffer, lookahead depth N+1, pace heuristic), memory-aware buffer (`MemoryBuffer.py` → RSS-based clamp, max buffer depth), and the pipelined compute driver (`PipelineDriver.py` → loads layer i, binds params, runs forward, evicts if needed).
- Use Python 3.14 + standard library + (aiohttp if installable; path to memory mmap via `mmap` stdlib).
- Produce a `README.md` per module + a minimal reproducible `examples/minimal_stream.py` that: builds a tiny 2-layer synthetic safetensors file, writes it to a local HTTP server with Range support, streams layer-by-layer, and prints TTFT + per-layer RSS. DO NOT skip this minimal example — it proves the pipeline works.
- Produce: `src/core/` source files + `examples/minimal_stream.py`
- FLAG honestly if torch/aiohttp/psutil are missing; state which packages must `pip install` and mark that as step-1 for engineer-bench.

### WS-3 — engineer-bench (storage + throttling + harness)
- Build in `src/storage/` + `src/bench/`: a local `RangeServer.py` (Python stdlib HTTP server serving safetensors bytes with HTTP Range requests + configurable simulated bandwidth throttle), a `BenchmarkHarness.py` that: spins the pipeline against monolithic baseline, measures TTFT, I/O masking ratio, end-to-end latency, RSS trajectory (psutil), and outputs a CSV + prints avg/percentiles. Provide a `run.sh` that kills the RangeServer and the runner, and a `setup.sh` that pip-installs missing deps (aiohttp, psutil, requests) and generates the 2-layer synthetic model.
- Produce: `src/storage/`, `src/bench/`, `run.sh`, `setup.sh`, `requirements.txt`
- NOTE: For a real first run in the next 24h, the single-node local RangeServer+HTTP-over-loopback is the pragmatic stand-in for MinIO + `tc`. Record this substitution explicitly in the final report (mention MinIO/tc as the production path).

### WS-4 — writer (docs/report/viva)
- Write the full EL project report skeleton: Title page, Abstract, Problem Statement, Literature Review (the 5 papers from the pasted docs — ServerlessLLM OSDI'24, LLM in a Flash, vLLM PagedAttention, FastServe, SOCK — cite correctly), Literature gap, Proposed architecture, Math model, Implementation plan (stack, modules, file list), Experimental methodology, Expected/placeholder results table, Publication viability (IEEE CloudCom/INDICON/DISCOVER/CONFLUENCE + the "your distinct contribution" table), Conclusion + Future Work.
- Write the Viva Voce Q&A for the 5 tough questions; keep the Q#1 and Q#2 answers realistic (if the compute can't fully hide I/O on CPU, say so plainly and quantify it). Include the deployment-vs-serverless reality check note.
- Produce: `docs/project_report.md` + `docs/viva_qa.md`

### WS-5 — lead (orchestrator check)
- Before starting any WS, confirm each specialist has the project context and the files it must produce (paths above).
- Dispatch the 4 specialists in parallel now.
- Set up the TODO file: `docs/build_todo.md` with the status of each WS.
- After specialists report, review each artifact for (a) file existence, (b) that code at least *parses/runs* the minimal example, (c) record the packages to install, and (d) produce the final consolidated `docs/PROJECT_SUBMISSION.md` (clean, submission-ready doc combining the report + math + architecture + viva + a short "status" note that these are real working artifacts).

## Constraints / honesty rules (do NOT violate)
- Do NOT fabricate benchmark results. If the pipeline has not run yet, say "TBD — pending `run.sh` execution" rather than inventing numbers.
- Do NOT overclaim memory savings: honestly state that full model params must be pre-allocated in the naive impl, so the win is TTFT + I/O masking, and the peak-RSS claim needs a custom param store (flag it as a known limitation).
- Do NOT claim 70%+ masking on CPU. Use the measured value; if masking is 30–60%, say so and explain why (layer compute on CPU is the bottleneck, not the network).
- Do NOT claim you learned a full model on CPU in 24h. Freeze models like SmolLM/TinyLlama are fine to download for the benchmark, but do not claim a specific latency number until it is measured.
- Each engineer must leave a `STATUS:` line in its file header stating: what is built, what is TBD, what packages to install.

## Now ACT
1. Write `docs/build_todo.md` (leading statuses: NOT STARTED → IN PROGRESS after dispatch).
2. Dispatch the four specialists (architect, engineer-core, engineer-bench, writer) in parallel. Reference their exact file/artifact responsibility above.
3. As each specialist reports, update TODO statuses and file headers.
4. When all four report, run a quick merge review and produce `docs/PROJECT_SUBMISSION.md`.
5. End by printing: (a) the file tree, (b) TODO statuses, (c) the one package install command, (d) the known-limitations list.
