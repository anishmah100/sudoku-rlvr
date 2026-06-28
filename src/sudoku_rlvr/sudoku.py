"""Pure-Python Sudoku core: generation, solving, uniqueness, (de)serialization.

Supports curriculum sizes 4x4, 6x6, 9x9. No third-party deps so it can be
unit-tested without the GPU stack installed.

Size -> (box_rows, box_cols):
  4 -> 2x2   6 -> 2x3 (2 tall, 3 wide)   9 -> 3x3
"""
from __future__ import annotations
import random
from typing import List, Optional, Tuple

Grid = List[List[int]]

BOX = {4: (2, 2), 6: (2, 3), 9: (3, 3)}

# Clue counts per size. Kept on the easier side so that exact solves actually
# occur during RL -- otherwise the +solve-bonus signal never fires and only the
# dense cell-accuracy term has gradient (which plateaus). More clues = easier.
DEFAULT_CLUES = {4: 9, 6: 24, 9: 52}


def _candidates(g: Grid, n: int, br: int, bc: int, r: int, c: int) -> List[int]:
    used = set()
    for i in range(n):
        used.add(g[r][i])
        used.add(g[i][c])
    r0, c0 = (r // br) * br, (c // bc) * bc
    for i in range(r0, r0 + br):
        for j in range(c0, c0 + bc):
            used.add(g[i][j])
    return [d for d in range(1, n + 1) if d not in used]


def generate_full(n: int, rng: random.Random) -> Grid:
    """A random complete valid grid via randomized backtracking."""
    br, bc = BOX[n]
    g = [[0] * n for _ in range(n)]

    def fill(pos: int) -> bool:
        if pos == n * n:
            return True
        r, c = divmod(pos, n)
        cands = _candidates(g, n, br, bc, r, c)
        rng.shuffle(cands)
        for d in cands:
            g[r][c] = d
            if fill(pos + 1):
                return True
        g[r][c] = 0
        return False

    fill(0)
    return g


def count_solutions(grid: Grid, n: int, limit: int = 2) -> int:
    """Count solutions up to `limit` using MRV backtracking (fast for uniqueness)."""
    br, bc = BOX[n]
    g = [row[:] for row in grid]
    count = 0

    def solve() -> None:
        nonlocal count
        best: Optional[Tuple[int, int]] = None
        best_c: Optional[List[int]] = None
        for r in range(n):
            for c in range(n):
                if g[r][c] == 0:
                    cs = _candidates(g, n, br, bc, r, c)
                    if not cs:
                        return  # dead end
                    if best_c is None or len(cs) < len(best_c):
                        best, best_c = (r, c), cs
        if best is None:
            count += 1
            return
        r, c = best
        for d in best_c:
            g[r][c] = d
            solve()
            g[r][c] = 0
            if count >= limit:
                return

    solve()
    return count


def make_puzzle(n: int, rng: random.Random, clues: Optional[int] = None) -> Tuple[Grid, Grid]:
    """Return (puzzle, solution). Removes cells while preserving a unique solution."""
    if clues is None:
        clues = DEFAULT_CLUES[n]
    solution = generate_full(n, rng)
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    holes_wanted = n * n - clues
    removed = 0
    for (r, c) in cells:
        if removed >= holes_wanted:
            break
        saved = puzzle[r][c]
        puzzle[r][c] = 0
        if count_solutions(puzzle, n, limit=2) != 1:
            puzzle[r][c] = saved  # would break uniqueness; keep as clue
        else:
            removed += 1
    return puzzle, solution


def to_str(grid: Grid, blank: str = ".") -> str:
    return "\n".join(" ".join(blank if v == 0 else str(v) for v in row) for row in grid)


def parse_grid(text: str, n: int) -> Optional[Grid]:
    """Parse the first n rows that each contain exactly n digits in 1..n.

    Tolerant of '.'/'0' blanks and extra whitespace. Returns None if no clean
    n x n block of fully-filled cells is found.
    """
    rows: List[List[int]] = []
    for line in text.splitlines():
        toks = line.replace(",", " ").split()
        vals: List[int] = []
        for t in toks:
            if t in (".", "0", "_"):
                vals.append(0)
            elif t.isdigit() and 1 <= int(t) <= n:
                vals.append(int(t))
            else:
                vals = []
                break
        if len(vals) == n:
            rows.append(vals)
            if len(rows) == n:
                return rows
    return None


def is_valid_solution(grid: Grid, n: int) -> bool:
    br, bc = BOX[n]
    full = set(range(1, n + 1))
    for r in range(n):
        if set(grid[r]) != full:
            return False
    for c in range(n):
        if set(grid[r][c] for r in range(n)) != full:
            return False
    for r0 in range(0, n, br):
        for c0 in range(0, n, bc):
            block = {grid[r0 + i][c0 + j] for i in range(br) for j in range(bc)}
            if block != full:
                return False
    return True
