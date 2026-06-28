#!/usr/bin/env python3
"""Measure the BASE model's solve rate as a function of difficulty (empty cells).

Answers: how easy must a puzzle be before Qwen2.5-3B can solve it at all? That
tells us where to start the difficulty curriculum and build up from.

    python scripts/difficulty_sweep.py --adapter none --tag base
    python scripts/difficulty_sweep.py --adapter outputs/adapter --tag trained
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sudoku_rlvr.data import build_examples  # noqa: E402
from sudoku_rlvr.modeling import generate, load_model  # noqa: E402
from sudoku_rlvr.rewards import score  # noqa: E402

RESULTS = os.path.join(ROOT, "results")

# (size, empties) grid to probe. clues = size*size - empties.
GRID = [
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 8),
    (6, 2), (6, 4), (6, 8), (6, 12),
    (9, 4), (9, 8), (9, 16),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/curriculum.yaml"))
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n", type=int, default=100, help="puzzles per difficulty")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tok = load_model(
        model_name=cfg["model"], max_seq_length=cfg["max_seq_length"],
        lora_rank=cfg["lora_rank"], gpu_memory_utilization=cfg["gpu_memory_utilization"],
        load_in_4bit=cfg.get("load_in_4bit", True), for_training=True,
    )
    lora = None if args.adapter == "none" else args.adapter

    rows = []
    for size, empties in GRID:
        clues = size * size - empties
        exs = build_examples(size, args.n, seed=777, clues=clues)
        # keep only those that actually hit the requested clue count (uniqueness
        # can force a few extra); difficulty label is the realized empties.
        chats = [e["prompt"] for e in exs]
        maxtok = 256 if size == 4 else (512 if size == 6 else 1024)
        texts = generate(model, tok, chats, maxtok, temperature=0.0, lora_path=lora)
        sc = [score(t, e["puzzle"], e["solution"], size) for t, e in zip(texts, exs)]
        realized = sum(e["puzzle"].count("0") for e in exs) / len(exs)
        solve = sum(s["solved"] for s in sc) / len(sc)
        frac = sum(s["frac_correct"] for s in sc) / len(sc)
        fmt = sum(s["format_ok"] for s in sc) / len(sc)
        rows.append({"size": size, "empties": empties, "realized_empties": round(realized, 1),
                     "solve_rate": solve, "frac_correct": frac, "format_rate": fmt})
        print(f"{size}x{size} empties={empties:2d} (clues {clues:2d}): "
              f"solve={solve:5.1%} frac={frac:5.1%} fmt={fmt:5.1%}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"difficulty_sweep_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
