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

import json
import random
from typing import Dict, List, Optional

from .prompts import build_chat
from .sudoku import DEFAULT_CLUES, make_puzzle, to_str


def build_examples(n: int, count: int, seed: int = 0,
                   clues: Optional[int] = None, cot: bool = True) -> List[Dict]:
    rng = random.Random(seed)
    if clues is None:
        clues = DEFAULT_CLUES[n]
    out: List[Dict] = []
    seen = set()
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        puzzle, solution = make_puzzle(n, rng, clues=clues)
        key = to_str(puzzle, blank="0")
        if key in seen:  # avoid duplicate puzzles
            continue
        seen.add(key)
        out.append({
            "prompt": build_chat(puzzle, n, cot=cot),
            "puzzle": to_str(puzzle, blank="0"),
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
