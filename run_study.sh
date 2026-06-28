#!/usr/bin/env bash
# End-to-end study: baseline eval -> curriculum GRPO -> post eval -> figures.
# Assumes setup.sh already created ./.venv. Safe to re-run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$HERE"
# shellcheck disable=SC1091
source .venv/bin/activate
source env.sh

mkdir -p data results assets outputs
[ -f data/eval_9x9.jsonl ] || python scripts/make_data.py

echo "===== BASELINE (before) ====="
python scripts/evaluate.py --adapter none --tag base

echo "===== TRAIN (curriculum GRPO) ====="
python scripts/train.py --config configs/curriculum.yaml

echo "===== POST-TRAIN (after) ====="
python scripts/evaluate.py --adapter outputs/adapter --tag trained

echo "===== FIGURES + TABLES ====="
python scripts/plot.py
echo "study complete."
