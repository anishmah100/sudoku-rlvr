"""Prompt construction for the Sudoku RLVR task.

We use a conversational format (a single user turn). The model is asked to
reason briefly, then emit the completed grid inside <answer></answer> tags so
the reward function can parse it deterministically.
"""
from __future__ import annotations

from .sudoku import BOX, Grid, to_str

SYSTEM = (
    "You are an expert Sudoku solver. You reason carefully and concisely, then "
    "give a final answer."
)
SYSTEM_DIRECT = (
    "You are an expert Sudoku solver. You output only the completed grid."
)

_RULES = (
    "Sudoku rules: fill every empty cell (shown as '.') with a digit from 1 to {n} "
    "so that each row, each column, and each {br}x{bc} box contains every digit "
    "from 1 to {n} exactly once. Each puzzle has a unique solution."
)

_INSTRUCTIONS = (
    "Think step by step, but keep your reasoning brief. When you are done, output "
    "the fully completed grid as {n} lines of {n} space-separated digits inside "
    "<answer> and </answer> tags. Do not include '.' in your answer; every cell "
    "must be filled.\n\n"
    "Example answer format for a {n}x{n} grid:\n"
    "<answer>\n{example}\n</answer>"
)

# Direct mode: no chain-of-thought. The weak 3B base confabulates when it reasons
# (and the rambling corrupts/truncates the grid), so for bootstrapping we ask it to
# copy the givens and fill the blanks directly.
_INSTRUCTIONS_DIRECT = (
    "Output ONLY the fully completed grid as {n} lines of {n} space-separated "
    "digits inside <answer> and </answer> tags. Copy the given digits exactly and "
    "fill every '.' so all constraints hold. Do not write any explanation.\n\n"
    "Answer format for a {n}x{n} grid:\n"
    "<answer>\n{example}\n</answer>"
)


def _example_block(n: int) -> str:
    # A purely illustrative (not necessarily valid) filled grid for formatting.
    rows = []
    for r in range(n):
        rows.append(" ".join(str((r + c) % n + 1) for c in range(n)))
    return "\n".join(rows)


def build_user_prompt(puzzle: Grid, n: int, cot: bool = True) -> str:
    br, bc = BOX[n]
    rules = _RULES.format(n=n, br=br, bc=bc)
    template = _INSTRUCTIONS if cot else _INSTRUCTIONS_DIRECT
    instr = template.format(n=n, example=_example_block(n))
    grid = to_str(puzzle, blank=".")
    return f"{rules}\n\nHere is the puzzle ({n}x{n}):\n\n{grid}\n\n{instr}"


def build_chat(puzzle: Grid, n: int, cot: bool = True) -> list[dict]:
    """Return a conversational prompt (list of role/content messages).

    cot=True  -> brief chain-of-thought then <answer>.
    cot=False -> direct grid only (better for the weak base; see module docstring).
    """
    return [
        {"role": "system", "content": SYSTEM if cot else SYSTEM_DIRECT},
        {"role": "user", "content": build_user_prompt(puzzle, n, cot=cot)},
    ]
