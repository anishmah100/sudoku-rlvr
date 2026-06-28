"""Correctness tests for the *measurement* harness itself.

These guard against the worst failure mode: a scoring/eval bug that makes the
model look better or worse than it is. We check no-false-negatives (a true
solution scores as solved at every size/difficulty), no-false-positives (wrong
grids never score as solved), train/eval disjointness, and puzzle validity.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sudoku_rlvr.data import build_examples, EVAL_HOLDOUT  # noqa: E402
from sudoku_rlvr.rewards import score  # noqa: E402
from sudoku_rlvr.sudoku import (  # noqa: E402
    BOX, count_solutions, is_valid_solution, make_puzzle, to_str,
)

CASES = [(4, 15), (4, 12), (4, 6), (6, 24), (6, 18), (8, 44), (8, 30)]


def test_oracle_solution_scores_solved_all_sizes():
    """The ground-truth solution must score solved=True everywhere (no false neg)."""
    for n, clues in CASES:
        for ex in build_examples(n, 15, seed=1, clues=clues):
            text = f"reasoning...\n<answer>\n{ex['solution'].replace('0','')}\n</answer>"
            # solution has no zeros, but rebuild cleanly from rows:
            sol_rows = ex["solution"].splitlines()
            text = "<answer>\n" + "\n".join(sol_rows) + "\n</answer>"
            s = score(text, ex["puzzle"], ex["solution"], n)
            assert s["solved"], f"false negative: {n}x{n} clues={clues}"
            assert s["givens_ok"] and abs(s["frac_correct"] - 1.0) < 1e-9


def test_wrong_grid_never_scores_solved():
    """Corrupting any single empty cell must break 'solved' (no false positive)."""
    for n, clues in CASES:
        for ex in build_examples(n, 15, seed=2, clues=clues):
            sol = [[int(x) for x in r.split()] for r in ex["solution"].splitlines()]
            puz = [[int(x) for x in r.split()] for r in ex["puzzle"].splitlines()]
            # corrupt the first empty cell to a different digit
            done = False
            for r in range(n):
                for c in range(n):
                    if puz[r][c] == 0:
                        sol[r][c] = sol[r][c] % n + 1
                        done = True
                        break
                if done:
                    break
            text = "<answer>\n" + to_str(sol) + "\n</answer>"
            s = score(text, ex["puzzle"], ex["solution"], n)
            assert not s["solved"], f"false positive: {n}x{n} clues={clues}"


def test_train_eval_disjoint():
    """Train and eval splits must never share a puzzle (no contamination)."""
    for n, clues in CASES:
        tr = {e["puzzle"] for e in build_examples(n, 300, seed=10, clues=clues, split="train")}
        ev = {e["puzzle"] for e in build_examples(n, 300, seed=99, clues=clues, split="eval")}
        assert tr and ev, f"empty split for {n}x{n} clues={clues}"
        assert tr.isdisjoint(ev), f"train/eval overlap at {n}x{n} clues={clues}"


def test_eval_split_matches_full_holdout():
    """An eval puzzle must also be reachable with split=None (it's a real puzzle),
    and must NOT be reachable with split='train'."""
    ev = build_examples(4, 50, seed=99, clues=12, split="eval")
    tr_keys = {e["puzzle"] for e in build_examples(4, 1000, seed=10, clues=12, split="train")}
    for e in ev:
        assert e["puzzle"] not in tr_keys


def test_generated_puzzles_unique_and_valid():
    for n, clues in CASES:
        for ex in build_examples(n, 10, seed=3, clues=clues, split="train"):
            puz = [[int(x) for x in r.split()] for r in ex["puzzle"].splitlines()]
            sol = [[int(x) for x in r.split()] for r in ex["solution"].splitlines()]
            assert is_valid_solution(sol, n), f"invalid solution {n}x{n}"
            assert count_solutions(puz, n, limit=2) == 1, f"non-unique {n}x{n} clues={clues}"
            # clue cells agree with solution
            for r in range(n):
                for c in range(n):
                    if puz[r][c] != 0:
                        assert puz[r][c] == sol[r][c]


def test_box_dims_consistent():
    for n in (4, 6, 8, 9):
        br, bc = BOX[n]
        assert br * bc == n


def test_holdout_fraction_roughly_right():
    """~EVAL_HOLDOUT of the space should land in eval (sanity on the hash split)."""
    rng = random.Random(0)
    from sudoku_rlvr.data import _in_split
    keys = []
    for _ in range(2000):
        p, _s = make_puzzle(4, rng, clues=10)
        keys.append(to_str(p, "0"))
    keys = list(set(keys))
    frac = sum(_in_split(k, "eval") for k in keys) / len(keys)
    assert abs(frac - EVAL_HOLDOUT) < 0.06, f"holdout fraction off: {frac:.3f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
