#!/usr/bin/env python3
"""Solve rate vs difficulty (empty cells) for a model — base or a LoRA adapter.

The same fixed grid is used for every run so results are directly comparable across
experiments. Writes per-cell metrics + a couple of sample completions.

    python scripts/difficulty_sweep.py --adapter none --out results/difficulty_sweep_base.json
    python scripts/difficulty_sweep.py --adapter experiments/exp001/adapter --out experiments/exp001/sweep.json
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

# Fixed, comprehensive (size, empties) grid. 4x4 fine, 6x6 medium, 8x8 coarse.
DEFAULT_GRID = (
    [(4, e) for e in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] +
    [(6, e) for e in (2, 4, 6, 8, 10, 12, 15, 18)] +
    [(8, e) for e in (4, 8, 12, 16, 20, 24, 30)]
)


def parse_grid_arg(s):
    if not s:
        return DEFAULT_GRID
    out = []
    for tok in s.split(","):
        a, b = tok.split(":")
        out.append((int(a), int(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/easy_curriculum.yaml"))
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--tag", default="base", help="used if --out not given")
    ap.add_argument("--n", type=int, default=100, help="puzzles per difficulty")
    ap.add_argument("--grid", default="", help="override grid, e.g. '4:8,8:20'")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    grid = parse_grid_arg(args.grid)

    model, tok = load_model(
        model_name=cfg["model"], max_seq_length=cfg["max_seq_length"],
        lora_rank=cfg["lora_rank"], gpu_memory_utilization=cfg["gpu_memory_utilization"],
        load_in_4bit=cfg.get("load_in_4bit", True), for_training=True,
    )
    lora = None if args.adapter == "none" else os.path.abspath(args.adapter)
    cot = cfg.get("cot", True)

    rows = []
    for size, empties in grid:
        clues = size * size - empties
        exs = build_examples(size, args.n, seed=args.seed, clues=clues, cot=cot,
                             split="eval")
        chats = [e["prompt"] for e in exs]
        maxtok = 512 if size == 4 else (768 if size == 6 else 1024)
        texts = generate(model, tok, chats, maxtok, temperature=0.0, lora_path=lora)
        sc = [score(t, e["puzzle"], e["solution"], size) for t, e in zip(texts, exs)]
        k = len(sc)
        rec = {
            "size": size, "empties": empties, "clues": clues,
            "realized_empties": round(sum(e["puzzle"].count("0") for e in exs) / k, 2),
            "solve_rate": sum(s["solved"] for s in sc) / k,
            "frac_correct": sum(s["frac_correct"] for s in sc) / k,
            "format_rate": sum(s["format_ok"] for s in sc) / k,
            "givens_ok_rate": sum(s["givens_ok"] for s in sc) / k,
            "mean_reward": sum(s["reward"] for s in sc) / k,
            "n": k,
            "sample": {"prompt": chats[0][-1]["content"], "completion": texts[0][:1500],
                       "solved": sc[0]["solved"]},
        }
        rows.append(rec)
        print(f"{size}x{size} empty={empties:2d} (clues {clues:2d}): "
              f"solve={rec['solve_rate']:5.1%} frac={rec['frac_correct']:5.1%} "
              f"fmt={rec['format_rate']:5.1%} givens={rec['givens_ok_rate']:5.1%}")

    out = args.out or os.path.join(RESULTS, f"difficulty_sweep_{args.tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
