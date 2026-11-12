#!/usr/bin/env bash
# STATUS: lead merge-review verifier. Built: file/compile/minimal-example checks.
# TBD: full benchmark run depends on all workstreams landing.
# Packages: none (uses python3/bash/curl already present; torch/numpy NOT required).
set -uo pipefail

cd "$(dirname "$0")/.."

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1
fail=0

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  [OK]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=$((fail+1)); }
warn() { printf '  [WARN] %s\n' "$*"; }

say "1. Python"
python3 --version || bad "python3 missing"

say "2. Required artifacts"
for f in \
  docs/01_architecture.md docs/02_math_model.md docs/project_report.md docs/viva_qa.md \
  docs/PROJECT_SUBMISSION.md docs/build_todo.md \
  src/core/__init__.py src/core/SafetensorsManifest.py src/core/S3ManifestWriter.py \
  src/core/AsyncPrefetcher.py src/core/MemoryBuffer.py src/core/PipelineDriver.py \
  src/core/README.md examples/minimal_stream.py \
  src/storage/__init__.py src/storage/RangeServer.py \
  src/bench/__init__.py src/bench/BenchmarkHarness.py \
  run.sh setup.sh requirements.txt
do
  [[ -s "$f" ]] && ok "$f" || bad "missing/empty: $f"
done

if [[ -s results/benchmark.csv ]] && head -n 1 results/benchmark.csv | grep -q 'first_layer_s'; then
  ok "results/benchmark.csv (with first_layer_s)"
else
  warn "results/benchmark.csv missing or lacks first_layer_s (run bash run.sh)"
fi

say "3. STATUS lines"
for f in $(find src examples -name '*.py' 2>/dev/null) run.sh setup.sh; do
  if grep -qi 'STATUS:' "$f" 2>/dev/null; then ok "STATUS in $f"; else bad "no STATUS: in $f"; fi
done

say "4. Python syntax"
if python3 -m py_compile src/core/*.py src/storage/*.py src/bench/*.py examples/*.py 2>/tmp/verify_pycompile.err; then
  ok "all Python files compile"
else
  bad "py_compile failed"; cat /tmp/verify_pycompile.err
fi

say "5. Shell syntax"
for s in run.sh setup.sh; do
  if bash -n "$s" 2>/tmp/verify_sh.err; then ok "$s parses"; else bad "$s syntax error"; cat /tmp/verify_sh.err; fi
done

say "6. Minimal streaming example (real run)"
if [[ -f examples/minimal_stream.py ]]; then
  if timeout 180 python3 examples/minimal_stream.py > /tmp/verify_minimal.log 2>&1; then
    ok "minimal_stream.py exited 0"
    tail -n 20 /tmp/verify_minimal.log
  else
    bad "minimal_stream.py failed (see /tmp/verify_minimal.log)"
    tail -n 30 /tmp/verify_minimal.log
  fi
else
  warn "skipped: examples/minimal_stream.py absent"
fi

if [[ "$FULL" == "1" ]]; then
  say "7. Full benchmark (setup.sh + run.sh)"
  if bash setup.sh && bash run.sh; then
    ok "benchmark run completed"
    if [[ -s results/benchmark.csv ]]; then
      ok "results/benchmark.csv written"
      head -n 10 results/benchmark.csv
    else
      bad "no results/benchmark.csv"
    fi
  else
    bad "setup/run failed"
  fi
else
  say "7. Full benchmark"
  warn "skipped (pass --full to run setup.sh + run.sh)"
fi

say "Summary"
if [[ "$fail" -eq 0 ]]; then
  printf '  ALL CHECKS PASSED\n'
  exit 0
else
  printf '  %d CHECK(S) FAILED\n' "$fail"
  exit 1
fi
