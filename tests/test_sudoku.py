import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sudoku_rlvr.sudoku import (  # noqa: E402
    BOX, count_solutions, generate_full, is_valid_solution, make_puzzle,
    parse_grid, to_str,
)


def test_generate_full_is_valid():
    rng = random.Random(0)
    for n in (4, 6, 9):
        g = generate_full(n, rng)
        assert is_valid_solution(g, n), f"invalid full grid for n={n}"


def test_make_puzzle_unique_and_consistent():
    rng = random.Random(1)
    for n in (4, 6, 9):
        puzzle, solution = make_puzzle(n, rng)
        assert is_valid_solution(solution, n)
        assert count_solutions(puzzle, n, limit=2) == 1, f"non-unique n={n}"
        # every clue agrees with the solution
        for r in range(n):
            for c in range(n):
                if puzzle[r][c] != 0:
                    assert puzzle[r][c] == solution[r][c]


def test_clue_count_respected():
    rng = random.Random(2)
    puzzle, _ = make_puzzle(9, rng, clues=40)
    given = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)
    # uniqueness constraint may keep a few extra clues, never fewer
    assert given >= 40


def test_parse_roundtrip():
    rng = random.Random(3)
    g = generate_full(9, rng)
    assert parse_grid(to_str(g), 9) == g


def test_parse_rejects_garbage():
    assert parse_grid("hello world", 4) is None
    assert parse_grid("1 2 3", 4) is None  # wrong width


def test_box_dims():
    assert BOX == {4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3)}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
