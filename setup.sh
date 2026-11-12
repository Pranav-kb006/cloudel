#!/usr/bin/env bash
# STATUS:
#   BUILT: idempotent environment setup — installs only the declared missing
#          Python packages (aiohttp, psutil, requests), generates the minimal
#          2-layer synthetic safetensors model AND the 8-layer benchmark model
#          if absent/invalid, and verifies parsing.
#   TBD:   real model download / MinIO upload (production path) is not part of
#          this setup; torch/numpy/safetensors are deliberately NOT required.
#   PACKAGES: aiohttp>=3.9, psutil>=5.9, requests>=2.31.
#
# Usage: bash setup.sh
# Overrides: PYTHON=python3 MODEL_2LAYER=... MODEL_BENCH=... MODEL_MB=8

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
MODEL_2LAYER="${MODEL_2LAYER:-models/synthetic_2layer.safetensors}"
MODEL_BENCH="${MODEL_BENCH:-models/synthetic_8layer.safetensors}"
MODEL_MB="${MODEL_MB:-8}"
BENCH_LAYERS="${BENCH_LAYERS:-8}"

echo "==> Python: $("${PYTHON}" --version 2>&1)"

# --- install only missing imports -----------------------------------------
MISSING=()
check_import() {
    local package="$1"
    local module="$2"
    if "${PYTHON}" -c "import ${module}" >/dev/null 2>&1; then
        echo "    present: ${package}"
    else
        echo "    MISSING: ${package}"
        MISSING+=("${package}")
    fi
}

echo "==> Checking declared dependencies"
check_import "aiohttp" "aiohttp"
check_import "psutil" "psutil"
check_import "requests" "requests"

if ((${#MISSING[@]} > 0)); then
    echo "==> Installing missing packages: ${MISSING[*]}"
    if ! "${PYTHON}" -m pip install --user -r requirements.txt; then
        echo "    'pip install --user' failed; retrying with --break-system-packages (user site)."
        "${PYTHON}" -m pip install --user --break-system-packages -r requirements.txt
    fi
else
    echo "==> All declared dependencies already installed (nothing to do)."
fi

# Verify the imports that the harness/core rely on.
"${PYTHON}" - <<'PY'
import importlib
import sys
missing = []
for module in ("psutil", "requests", "aiohttp"):
    try:
        mod = importlib.import_module(module)
        print(f"    ok: {module} {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{module}: {exc}")
if missing:
    print("ERROR: dependencies still missing:", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    raise SystemExit(1)
PY

# --- generate synthetic models if absent or invalid -----------------------
mkdir -p models results

ensure_model() {
    local path="$1"
    local layers="$2"
    local label="$3"
    local ok=0
    if [[ -s "${path}" ]]; then
        if "${PYTHON}" -c "
import sys
from src.storage.safetensors_ranges import parse_safetensors
m = parse_safetensors(sys.argv[1])
assert m['layers'], 'no layers'
print(f\"    existing model ok: {m['size']} bytes, layers={list(m['layers'])}\")
" "${path}"; then
            ok=1
        else
            echo "    existing model is invalid; regenerating."
        fi
    fi
    if [[ "${ok}" -eq 0 ]]; then
        echo "==> Generating ${label}: ${path} (~${MODEL_MB} MiB, ${layers} layers)"
        "${PYTHON}" -m src.bench.make_synthetic_model \
            --out "${path}" --layers "${layers}" --size-mb "${MODEL_MB}" --seed 1234
    else
        echo "==> ${label} already present (idempotent skip): ${path}"
    fi
}

ensure_model "${MODEL_2LAYER}" 2 "minimal 2-layer model"
ensure_model "${MODEL_BENCH}" "${BENCH_LAYERS}" "benchmark model"

# --- final verification ----------------------------------------------------
echo "==> Verifying model ranges (stdlib parser)"
for model in "${MODEL_2LAYER}" "${MODEL_BENCH}"; do
    "${PYTHON}" -c "
import sys
from src.storage.safetensors_ranges import parse_safetensors
m = parse_safetensors(sys.argv[1])
assert m['size'] > 0
for name, meta in m['layers'].items():
    assert 0 <= meta['start'] < meta['end'] <= m['size'], (name, meta)
print(f\"    {m['file']}: {m['size']} bytes, {len(m['layers'])} layers, {len(m['tensors'])} tensors\")
for name, meta in m['layers'].items():
    print(f\"      {name:<10} [{meta['start']}:{meta['end']}] length={meta['length']}\")
" "${model}"
done

echo "==> Setup complete."
echo "    Next: bash run.sh   (benchmarks ${MODEL_BENCH})"
echo "    Minimal artifact: ${MODEL_2LAYER}"
echo "    Output: results/benchmark.csv (or an explicit TBD row if no run succeeds)."
