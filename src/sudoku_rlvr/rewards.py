"""Verifiable rewards for Sudoku RLVR.

The reward is a pure function of the model's text and the (known, unique)
ground-truth solution -- exactly the property that makes this an RLVR task.

Scoring (per completion):
  format_ok    : a clean n x n block of digits 1..n parsed from <answer>..</answer>
  givens_ok    : all original clues preserved (model did not overwrite givens)
  frac_correct : fraction of empty cells that match the unique solution
  solved       : the parsed grid equals the solution exactly

Shaped scalar reward (dense, so GRPO sees within-group variance early):
  +0.20  parse/format
  -0.50  givens violated (an invalid attempt)
  +1.00 * frac_correct
  +2.00  exact solve bonus
"""
from __future__ import annotations

import re
from typing import List, Optional

from .sudoku import Grid, parse_grid

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

W_FORMAT = 0.20
W_GIVENS_VIOLATION = -0.50
W_FRAC = 1.00
W_SOLVE = 2.00


def _completion_text(completion) -> str:
    """Accept either a raw string or a conversational [{'role','content'}, ...]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def extract_answer(text: str, n: int) -> Optional[Grid]:
    """Pull the last <answer> block and parse an n x n grid from it.

    Falls back to parsing the whole text if no tags are present.
    """
    blocks = _ANSWER_RE.findall(text)
    candidates = blocks[-1:] if blocks else [text]
    for block in candidates:
        g = parse_grid(block, n)
        if g is not None:
            return g
    return None


def _to_grid(s) -> Grid:
    if isinstance(s, str):
        return [[int(x) for x in row.split()] for row in s.strip().splitlines()]
    return s


def score(text: str, puzzle, solution, n: int) -> dict:
    """Full scoring breakdown for one completion. Used by training and eval."""
    puzzle = _to_grid(puzzle)
    solution = _to_grid(solution)
    out = {
        "format_ok": False,
        "givens_ok": False,
        "frac_correct": 0.0,
        "solved": False,
        "reward": 0.0,
    }
    grid = extract_answer(text, n)
    if grid is None:
        return out
    out["format_ok"] = True
    reward = W_FORMAT

    givens_ok = True
    empties = correct = 0
    for r in range(n):
        for c in range(n):
            if puzzle[r][c] != 0:
                if grid[r][c] != puzzle[r][c]:
                    givens_ok = False
            else:
                empties += 1
                if grid[r][c] == solution[r][c]:
                    correct += 1
    out["givens_ok"] = givens_ok
    out["frac_correct"] = (correct / empties) if empties else 1.0
    out["solved"] = givens_ok and grid == solution

    if not givens_ok:
        reward += W_GIVENS_VIOLATION
    reward += W_FRAC * out["frac_correct"]
    if out["solved"]:
        reward += W_SOLVE
    out["reward"] = reward
    return out


# ---- TRL GRPO reward functions --------------------------------------------
# TRL calls each reward fn with (prompts, completions, **columns) where the
# extra dataset columns (puzzle, solution, size) arrive as parallel lists. TRL
# sums the outputs of all reward functions. We split into two so each shows up
# separately in the logged metrics.

def _sizes(size, n_default, batch):
    if size is None:
        return [n_default] * batch
    return [int(s) for s in size]


def format_reward(completions, puzzle=None, solution=None, size=None, **kw) -> List[float]:
    texts = [_completion_text(c) for c in completions]
    sizes = _sizes(size, 9, len(texts))
    return [W_FORMAT if extract_answer(t, n) is not None else 0.0
            for t, n in zip(texts, sizes)]


def solution_reward(completions, puzzle=None, solution=None, size=None, **kw) -> List[float]:
    texts = [_completion_text(c) for c in completions]
    sizes = _sizes(size, 9, len(texts))
    rewards = []
    for i, (t, n) in enumerate(zip(texts, sizes)):
        s = score(t, puzzle[i], solution[i], n)
        # subtract the format component so it is not double-counted with format_reward
        rewards.append(s["reward"] - (W_FORMAT if s["format_ok"] else 0.0))
    return rewards


def solved_metric(completions, puzzle=None, solution=None, size=None, **kw) -> List[float]:
    """A 0/1 'fully solved' signal -- weight 0 in training, used only for logging."""
    texts = [_completion_text(c) for c in completions]
    sizes = _sizes(size, 9, len(texts))
    return [1.0 if score(t, puzzle[i], solution[i], n)["solved"] else 0.0
            for i, (t, n) in enumerate(zip(texts, sizes))]


def get_reward_funcs():
    return [format_reward, solution_reward]
