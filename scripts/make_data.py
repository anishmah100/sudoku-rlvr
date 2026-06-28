#!/usr/bin/env python3
"""Generate curriculum datasets (train + eval) as JSONL under ./data.

Usage:
    python scripts/make_data.py                     # default sizes/counts
    python scripts/make_data.py --train 1500 --eval 200
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sudoku_rlvr.data import build_examples, write_jsonl  # noqa: E402
from sudoku_rlvr.sudoku import DEFAULT_CLUES  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 9])
    ap.add_argument("--train", type=int, default=1500, help="train puzzles per size")
    ap.add_argument("--eval", type=int, default=200, help="eval puzzles per size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    for n in args.sizes:
        for split, count, seed in (("train", args.train, args.seed),
                                   ("eval", args.eval, args.seed + 10_000)):
            t0 = time.time()
            rows = build_examples(n, count, seed=seed, clues=DEFAULT_CLUES[n])
            path = os.path.join(DATA, f"{split}_{n}x{n}.jsonl")
            write_jsonl(path, rows)
            dt = time.time() - t0
            print(f"{split:5s} {n}x{n}: {len(rows):5d} puzzles "
                  f"({DEFAULT_CLUES[n]} clues) -> {path}  [{dt:.1f}s]")


if __name__ == "__main__":
    main()
