import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sudoku_rlvr.rewards import (  # noqa: E402
    extract_answer, format_reward, score, solution_reward, solved_metric,
)
from sudoku_rlvr.sudoku import make_puzzle, to_str  # noqa: E402


def _case(n=4, seed=0):
    rng = random.Random(seed)
    puzzle, solution = make_puzzle(n, rng)
    return puzzle, solution, n


def test_perfect_solution_scores_max():
    puzzle, solution, n = _case()
    text = f"reasoning...\n<answer>\n{to_str(solution)}\n</answer>"
    s = score(text, to_str(puzzle, '0'), to_str(solution, '0'), n)
    assert s["solved"] and s["givens_ok"] and s["format_ok"]
    assert abs(s["frac_correct"] - 1.0) < 1e-9
    assert s["reward"] > 3.0  # 0.2 + 1.0 + 2.0


def test_unparseable_scores_zero():
    puzzle, solution, n = _case()
    s = score("I cannot solve this.", to_str(puzzle, '0'), to_str(solution, '0'), n)
    assert not s["format_ok"] and s["reward"] == 0.0


def test_partial_is_between():
    puzzle, solution, n = _case(seed=5)
    grid = [row[:] for row in solution]
    # corrupt one empty cell
    for r in range(n):
        for c in range(n):
            if puzzle[r][c] == 0:
                grid[r][c] = grid[r][c] % n + 1
                break
        else:
            continue
        break
    text = f"<answer>\n{to_str(grid)}\n</answer>"
    s = score(text, to_str(puzzle, '0'), to_str(solution, '0'), n)
    assert not s["solved"]
    assert 0.0 < s["frac_correct"] < 1.0


def test_givens_violation_penalized():
    puzzle, solution, n = _case(seed=7)
    grid = [row[:] for row in solution]
    for r in range(n):
        for c in range(n):
            if puzzle[r][c] != 0:
                grid[r][c] = puzzle[r][c] % n + 1  # overwrite a given
                break
        else:
            continue
        break
    text = f"<answer>\n{to_str(grid)}\n</answer>"
    s = score(text, to_str(puzzle, '0'), to_str(solution, '0'), n)
    assert not s["givens_ok"] and not s["solved"]


def test_extract_uses_last_block():
    g = "1 2\n2 1"  # not parsed for n=4, just checks tag selection
    assert extract_answer("<answer>x</answer><answer>\n1 2 3 4\n2 1 4 3\n3 4 1 2\n4 3 2 1\n</answer>", 4) is not None


def test_trl_reward_funcs_batch():
    puzzle, solution, n = _case(seed=9)
    comps = [[{"role": "assistant", "content": f"<answer>\n{to_str(solution)}\n</answer>"}]]
    kw = dict(puzzle=[to_str(puzzle, '0')], solution=[to_str(solution, '0')], size=[n])
    assert format_reward(comps, **kw)[0] == 0.20
    assert solution_reward(comps, **kw)[0] > 2.5
    assert solved_metric(comps, **kw)[0] == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
