"""Dataset construction for the curriculum.

Each example is a dict with:
  prompt   : conversational messages (list of {role, content})
  puzzle   : the puzzle grid serialized as text rows
  solution : the unique solution serialized as text rows
  size     : board size (4, 6, or 9)
  clues    : number of givens

Stored as JSONL so generation needs no third-party deps; training loads it via
`datasets.load_dataset("json", ...)`.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Optional

from .prompts import build_chat
from .sudoku import DEFAULT_CLUES, make_puzzle, to_str

# Fraction of the puzzle space reserved for evaluation. The split is by a stable
# hash of the puzzle string, so train and eval sets are PROVABLY disjoint for any
# size/clue count and any seed -- critical for tiny spaces (e.g. 1-empty 4x4) where
# random train/eval seeds would otherwise overlap and inflate "after" numbers.
EVAL_HOLDOUT = 0.15
_HOLDOUT_BUCKETS = 1000


def _bucket(puzzle_key: str) -> int:
    h = hashlib.sha256(puzzle_key.encode()).hexdigest()
    return int(h[:8], 16) % _HOLDOUT_BUCKETS


def _in_split(puzzle_key: str, split: Optional[str]) -> bool:
    """split='train' -> non-holdout; split='eval' -> holdout; None -> everything."""
    if split is None:
        return True
    is_eval = _bucket(puzzle_key) < int(EVAL_HOLDOUT * _HOLDOUT_BUCKETS)
    return is_eval if split == "eval" else not is_eval


def build_examples(n: int, count: int, seed: int = 0,
                   clues: Optional[int] = None, cot: bool = True,
                   split: Optional[str] = None) -> List[Dict]:
    """Generate `count` unique puzzles.

    split: 'train' (non-holdout puzzles only), 'eval' (holdout only), or None (any).
    Train and eval splits are guaranteed disjoint by a hash of the puzzle.
    """
    rng = random.Random(seed)
    if clues is None:
        clues = DEFAULT_CLUES[n]
    out: List[Dict] = []
    seen = set()
    attempts = 0
    while len(out) < count and attempts < count * 200:
        attempts += 1
        puzzle, solution = make_puzzle(n, rng, clues=clues)
        key = to_str(puzzle, blank="0")
        if key in seen or not _in_split(key, split):
            continue
        seen.add(key)
        out.append({
            "prompt": build_chat(puzzle, n, cot=cot),
            "puzzle": key,
            "solution": to_str(solution, blank="0"),
            "size": n,
            "clues": clues,
        })
    return out


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
