# Methodology

The task definition, reward, curriculum, algorithm, and evaluation protocol.

## Task

An `n×n` Sudoku has box dimensions `(b_r, b_c)` with `b_r · b_c = n`:

| n | box | digits |
|---|-----|--------|
| 4 | 2×2 | 1–4 |
| 6 | 2×3 | 1–6 |
| 8 | 2×4 | 1–8 |
| 9 | 3×3 | 1–9 |

A puzzle is a partial assignment (the givens); a solution is a full assignment
satisfying the row, column, and box all-different constraints. `sudoku.py` draws a
complete grid by randomized backtracking, then removes cells in random order, keeping a
removal only if the puzzle still has exactly one solution (checked by an MRV-ordered
solver that counts up to two). Each puzzle therefore has a single correct grid, which
is the reward target.

## Reward

Let `G` be the parsed grid, `P` the puzzle, `S` the solution, and `E` the empty cells
of `P`:

```
format_ok  = G is a clean n×n block of digits 1..n inside <answer>…</answer>
givens_ok  = G[r,c] = P[r,c] for every clue cell (r,c)
frac       = |{ (r,c) in E : G[r,c] = S[r,c] }| / |E|
solved     = givens_ok and G = S

r = 0                          if not format_ok
  = 0.20                       format
    − 0.50 if not givens_ok    clue overwritten
    + 1.00 · frac              cell accuracy on empty cells
    + 2.00 if solved           exact match
```

GRPO standardizes rewards within each group of completions for a prompt. A binary
solved/not-solved reward is near-constant early in training and produces no gradient;
the `frac` term varies across completions in a group from the start. The reward is
implemented once (`rewards.score`) and used by both the trainer and the evaluator.

## Curriculum

Two configs are provided:

- `configs/curriculum.yaml` — size-based (4×4 → 6×6 → 9×9). The base model never solves
  6×6/9×9 during training, so the solve term never fires and the policy does not move.
  Kept as a reference for that result.
- `configs/easy_curriculum.yaml` — difficulty-based, 4×4 only, increasing the number of
  empty cells per stage with LoRA carried forward:

  | Stage | Empty cells | Clues | Steps |
  |---|---|---|---|
  | 0 | 1 | 15 | 120 |
  | 1 | 2 | 14 | 150 |
  | 2 | 3 | 13 | 150 |
  | 3 | 5 | 11 | 200 |
  | 4 | 7 |  9 | 250 |

The curriculum begins at a difficulty the base model solves with non-zero probability
(measured by `scripts/difficulty_sweep.py`), since GRPO can only reinforce completions
that already occur. Each stage's puzzles are generated in-process at the configured clue
count, drawn from the training split (see Evaluation).

## GRPO

Group Relative Policy Optimization (Shao et al., DeepSeekMath):

1. For a prompt, sample a group of `K` completions.
2. Score each with the reward.
3. Advantage `A_i = (r_i − mean(r)) / std(r)`, relative within the group.
4. Update toward high-advantage completions with a KL penalty to the reference policy.

There is no value network, which keeps memory within 24 GB alongside the vLLM engine.

Hyperparameters are set per config; the difficulty curriculum uses LoRA rank 32, `K`=8,
LR 1e-5 (cosine, warmup), adamw_8bit, train temperature 1.0, and
`gpu_memory_utilization` 0.6.

## Evaluation

- Train and eval puzzles are disjoint: `data.py` assigns each puzzle to a train or eval
  pool by a hash of its string (15% held out), independent of the random seed. This
  keeps the sets disjoint even for small spaces such as 4×4 with one empty cell.
- `scripts/difficulty_sweep.py` reports, per `(size, empty cells)` cell, the
  `solve_rate`, `frac_correct`, `format_rate`, `givens_ok_rate`, and `mean_reward` on
  100 held-out puzzles with greedy decoding, for either the base model or an adapter.
- Puzzles are generated at run time, so they cannot appear in any pretraining corpus.

## Tests

`tests/test_harness.py` checks that a ground-truth solution scores as solved at every
size and difficulty (no false negatives), that corrupted grids never score as solved
(no false positives), that train and eval splits are disjoint, and that generated
puzzles are valid and unique.

## Notes

- The reward checks only the final grid. A higher solve rate does not imply the
  intermediate reasoning is correct.
- The parser accepts `.`/`0`/`_` blanks, commas, and extra whitespace, so a reasonable
  output format is not penalized.
