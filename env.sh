#!/usr/bin/env bash
# Source this to keep ALL caches/state inside the project (no global writes).
#   source env.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="$HERE"
export HF_HOME="$HERE/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$HERE/.cache/hf/hub"
export TRANSFORMERS_CACHE="$HERE/.cache/hf/transformers"
export VLLM_CACHE_ROOT="$HERE/.cache/vllm"
export TRITON_CACHE_DIR="$HERE/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HERE/.cache/inductor"
export XDG_CACHE_HOME="$HERE/.cache/xdg"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
echo "[env] project-local caches under $HERE/.cache"
