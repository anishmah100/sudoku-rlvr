# Methodology

How the model is trained and evaluated, covering every technique used across the runs
in [EXPERIMENTS.md](EXPERIMENTS.md).

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
solver that counts up to two). Each puzzle therefore has a single correct grid, which is
the reward target. Difficulty is controlled by the number of empty cells.

## Prompt

A single user turn states the rules and the puzzle (with `.` for blanks) and asks for
brief reasoning followed by the completed grid inside `<answer>…</answer>`. Reasoning is
kept because the direct-answer variant (grid only) reaches 100% format but 0% solve —
the model fills cells only when it reasons first. A direct mode remains in `prompts.py`
for reference.

## Reward

Computed against the unique solution (`rewards.score`); `G` is the parsed grid, `P` the
puzzle, `S` the solution, `E` the empty cells of `P`:

```
format_ok  = G is a clean n×n block of digits 1..n inside <answer>…</answer>
givens_ok  = G[r,c] = P[r,c] for every clue cell
frac       = |{ (r,c) in E : G[r,c] = S[r,c] }| / |E|     (1.0 if E is empty)
solved     = givens_ok and G = S

r = 0                          if not format_ok
  = 0.20                       format
    − 0.50 if not givens_ok    a clue was overwritten
    + 1.00 · frac              cell accuracy on empty cells
    + 2.00 if solved           exact match
```

GRPO standardizes rewards within each group of completions for a prompt. A binary
solved/not-solved reward is near-constant early in training and gives no gradient; the
`frac` term varies across a group from the start. The same function is used by the
trainer and the evaluator, so training signal and reported metrics are identical.

## Curriculum

The central method is a **difficulty curriculum**: begin at a difficulty the base model
already solves with non-zero probability, then increase the number of empty cells one
stage at a time, carrying the LoRA weights forward. Each stage's puzzles are generated
in-process at the configured clue count (from the training split).

A **size-based curriculum** (4×4 → 6×6 → 9×9 at standard clue counts) was tried first
and does not work: the base model never produces a correct 6×6/9×9 grid, so the solve
term never fires and the policy does not move (KL to reference stays ~0.005). It is kept
as `configs/curriculum.yaml` for reference.

The governing constraint is that GRPO reinforces completions that already occur — it
cannot create behavior the policy never exhibits. `scripts/difficulty_sweep.py` is used
to find the difficulty where the base model's solve rate is non-zero, which is where the
curriculum starts.

## Techniques used across runs

- **Transcription warm-up (8×8).** On 8×8 the base model cannot reliably reproduce the
  64-cell grid (it preserves all givens in ~3% of completions), so no curriculum makes
  progress. A stage-0 task with 0 empty cells (copy the grid exactly) trains faithful
  transcription first; only then does difficulty training work. This raised 8×8 format
  from ~50% to ~100% and unlocked solving.
- **Resume / continued training.** A run can resume a prior adapter (`resume_adapter`).
  This is used to spend compute on harder difficulties without relearning easy ones, and
  to transfer a 4×4 adapter into 6×6 (which it already solves partially). The frontier
  keeps improving across successive resumed runs until it converges.
- **Stability.** Resuming a strong adapter with a high learning rate and single-prompt
  updates caused a KL spike and collapse in one run. The stable settings are gradient
  accumulation over several prompts (lower-variance updates), gradient clipping
  (`max_grad_norm` 0.5), and a smaller learning rate (5e-6 when resuming, 1e-5 from
  base).

## GRPO

Group Relative Policy Optimization (Shao et al., DeepSeekMath): sample a group of `K`
completions per prompt, score each with the reward, compute advantages relative within
the group (`A_i = (r_i − mean) / std`), and update toward high-advantage completions with
a KL penalty to the reference policy. There is no value network, which keeps memory
within 24 GB alongside the vLLM generation engine. Runs use `K`=8, LoRA rank 32,
4-bit Qwen2.5-3B-Instruct, adamw_8bit, train temperature 1.0, `gpu_memory_utilization`
0.55. Generation uses Unsloth's in-process vLLM engine (colocate), not a separate server.

## Evaluation

- **Disjoint train/eval.** Each puzzle is assigned to a train or eval pool by a hash of
  its string (`data.py`, 15% held out), independent of the random seed, so the two are
  disjoint even for small spaces (4×4 with one empty cell has only a few thousand
  puzzles). `tests/test_harness.py` verifies disjointness.
- **Metric.** `scripts/difficulty_sweep.py` reports, per `(size, empty cells)`, the
  exact-solve rate plus cell accuracy, format rate, givens-preserved rate, and mean
  reward, on 100 held-out puzzles with greedy decoding, for the base model or any
  adapter. The headline number is exact-solve rate.
- **Base comparison.** The same sweep is run on the base model (`--adapter none`) and on
  each trained adapter; `scripts/frontier.py` renders the base-vs-trained comparison.
- Puzzles are generated at run time and cannot appear in any pretraining corpus.

## Reproducibility and tests

- `bash run_experiment.sh <id> <config>` runs train → held-out sweep → charts →
  registry, writing everything to `experiments/<id>/`.
- `tests/test_harness.py` checks the measurement harness: a ground-truth solution scores
  as solved at every size/difficulty (no false negatives), corrupted grids never score
  as solved (no false positives), train/eval splits are disjoint, and generated puzzles
  are valid and unique. `tests/test_sudoku.py` and `tests/test_rewards.py` cover the
  generator and reward.

## Notes

- The reward checks only the final grid; a higher solve rate does not imply the
  intermediate reasoning is correct.
- The parser accepts `.`/`0`/`_` blanks, commas, and extra whitespace, so a reasonable
  output format is not penalized.
