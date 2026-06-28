#!/usr/bin/env bash
# Create a project-local virtualenv and install the GPU stack. Touches nothing
# global: the venv lives at ./.venv and all caches live under ./.cache.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
echo "[setup] creating venv at ./.venv with $($PY --version)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
source env.sh

python -m pip install --upgrade pip wheel
echo "[setup] installing requirements (this downloads torch/vLLM/unsloth ~ several GB)"
pip install -r requirements.txt

echo "[setup] sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "[setup] done. Activate with:  source .venv/bin/activate && source env.sh"
