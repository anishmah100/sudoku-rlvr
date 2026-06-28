#!/usr/bin/env bash
# "Start trivial, work up" 4x4 study: train the difficulty curriculum, then sweep
# the trained model across difficulties so we can chart the before/after climb.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$HERE"
source .venv/bin/activate
source env.sh

echo "===== TRAIN (4x4 trivial -> full) ====="
python scripts/train.py --config configs/easy_curriculum.yaml

echo "===== DIFFICULTY SWEEP (trained) ====="
python scripts/difficulty_sweep.py --adapter outputs/adapter --tag trained --n 100

echo "===== FIGURES ====="
python scripts/plot.py
echo "easy study complete."
