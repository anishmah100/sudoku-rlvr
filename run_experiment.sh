#!/usr/bin/env bash
# Run one fully-logged experiment: train -> held-out sweep -> charts -> registry.
#   bash run_experiment.sh <exp_id> <config.yaml>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$HERE"
source .venv/bin/activate
source env.sh

EXP_ID="${1:?usage: run_experiment.sh <exp_id> <config>}"
CONFIG="${2:?usage: run_experiment.sh <exp_id> <config>}"
EXP="experiments/$EXP_ID"
mkdir -p "$EXP"
cp "$CONFIG" "$EXP/config.yaml"

echo "===== TRAIN ($EXP_ID) ====="
python scripts/train.py --config "$CONFIG" --exp-dir "$EXP" || { echo "TRAIN FAILED"; exit 1; }

echo "===== HELD-OUT DIFFICULTY SWEEP ====="
python scripts/difficulty_sweep.py --config "$CONFIG" --adapter "$EXP/adapter" --out "$EXP/sweep.json" || { echo "SWEEP FAILED"; exit 1; }

echo "===== CHARTS + REGISTRY ====="
python scripts/plot.py --exp-dir "$EXP"
python scripts/log_experiment.py --exp-dir "$EXP"
echo "experiment $EXP_ID complete."
