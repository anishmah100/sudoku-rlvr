# sudoku-rlvr

Training Qwen2.5-3B to solve Sudoku with GRPO (reinforcement learning with a
verifiable reward) on a single RTX 4090. The reward is a pure function of the
generated grid against the puzzle's unique solution, so no reward model or judge is
involved.

<p align="center">
  <img src="assets/difficulty_sweep.png" width="98%" alt="Solve rate vs difficulty, base vs trained"/>
</p>

## Summary

The base model solves almost no Sudoku puzzles. After a difficulty curriculum on 4×4
boards, it solves 99% of 4×4 puzzles with one empty cell and 32% of full 4×4 puzzles,
and shows some transfer to 6×6 and 9×9 boards that were not trained on.

A size-based curriculum (4×4 → 6×6 → 9×9 at standard clue counts) does not work: the
base model never produces a correct 6×6 or 9×9 grid during training, so the solve
reward is never triggered and the policy does not move. The difficulty curriculum
below starts at a difficulty the base model can occasionally solve (4×4 with a single
empty cell) and increases the number of empty cells one stage at a time, carrying the
LoRA weights forward. That configuration is kept in `configs/curriculum.yaml` as a
reference for the negative result.

## Setup

| | |
|---|---|
| GPU | 1 × NVIDIA RTX 4090 (24 GB) |
| Model | Qwen2.5-3B-Instruct, 4-bit, LoRA rank 32 |
| Generation | vLLM, colocated in-process with training |
| Trainer | Unsloth + TRL `GRPOTrainer` |

The virtualenv, model and vLLM caches, and checkpoints all live under the project
directory (`.venv/`, `.cache/`).

## Method

**Task.** Each prompt contains an `n×n` puzzle (`.` for blanks) and asks for brief
reasoning followed by the completed grid inside `<answer>…</answer>`. Reasoning is
included because the direct-answer variant solves 0% — the model fills cells only when
it reasons first.

**Reward** ([`rewards.py`](src/sudoku_rlvr/rewards.py)), computed against the unique
solution:

| Term | Weight | Condition |
|---|---|---|
| format | +0.20 | a parseable `n×n` grid of digits in `<answer>` |
| givens violated | −0.50 | an original clue was overwritten |
| cell accuracy | +1.00 × *frac* | fraction of empty cells matching the solution |
| solve bonus | +2.00 | the grid equals the solution exactly |

The cell-accuracy term provides gradient before the model can fully solve a puzzle; the
solve bonus rewards exact completion.

**Curriculum** ([`configs/easy_curriculum.yaml`](configs/easy_curriculum.yaml)), 4×4
only, LoRA carried across stages, puzzles generated per stage:

| Stage | Empty cells | Clues | Steps |
|---|---|---|---|
| 0 | 1 | 15 | 120 |
| 1 | 2 | 14 | 150 |
| 2 | 3 | 13 | 150 |
| 3 | 5 | 11 | 200 |
| 4 | 7 | 9 | 250 |

**GRPO.** For each prompt, sample a group of completions, score each with the reward,
and update toward the above-average completions in the group. There is no value
network, which keeps memory within 24 GB alongside the vLLM engine.

**Train/eval split.** Puzzles are partitioned into train and eval pools by a hash of
the puzzle string ([`data.py`](src/sudoku_rlvr/data.py)), so the two are disjoint even
for small puzzle spaces such as 4×4 with one empty cell.

## Results

Figures and tables are produced by `scripts/plot.py` and
`scripts/log_experiment.py` from logged metrics. See
[`results/comparison.md`](results/comparison.md),
[`results/summary.csv`](results/summary.csv),
[`results/observations.md`](results/observations.md), and
[`docs/RESULTS.md`](docs/RESULTS.md).

Exact-solve rate on 100 held-out puzzles per cell, greedy decoding:

| Puzzle | Empty cells | Base | Trained |
|---|---|---|---|
| 4×4 | 1 | 7% | 99% |
| 4×4 | 2 | 0% | 93% |
| 4×4 | 3 | 3% | 86% |
| 4×4 | 4 | 0% | 73% |
| 4×4 | 6 | 0% | 46% |
| 4×4 | 8 | 0% | 32% |
| 6×6 | 2 | 0% | 17% |
| 9×9 | 4 | 4% | 19% |

Training touched only 4×4 boards; the 6×6 and 9×9 rows are zero-shot.

<p align="center"><img src="assets/training_curves.png" width="97%" alt="Training metrics per stage"/></p>

## Reproduce

```bash
cd sudoku-rlvr
bash setup.sh
source .venv/bin/activate && source env.sh

python scripts/difficulty_sweep.py --adapter none --out results/difficulty_sweep_base.json
bash run_easy.sh
```

Core tests (no GPU): `python tests/test_sudoku.py && python tests/test_rewards.py && python tests/test_harness.py`

## Layout

```
src/sudoku_rlvr/
  sudoku.py      puzzle generation, MRV uniqueness solver, serialization
  prompts.py     prompt construction (CoT and direct modes)
  rewards.py     verifiable reward and TRL reward functions
  data.py        per-difficulty dataset builder with hash-based train/eval split
  modeling.py    model load + generation (Unsloth/vLLM)
scripts/
  train.py            curriculum GRPO with per-experiment logging
  difficulty_sweep.py solve rate vs difficulty for a base model or adapter
  evaluate.py         fixed-set evaluation
  plot.py             charts
  log_experiment.py   per-experiment summary + registry entry
  monitor.py          training-health check from a run log
configs/    curriculum configs
tests/      unit tests for the core and the measurement harness
docs/       METHODOLOGY.md, RESULTS.md, EXPERIMENTS.md
experiments/ per-run logs, charts, summaries, REGISTRY.md
```

## Limitations

- The trained model is specialized for this task; it is not a stronger general
  reasoner than the base model.
- RLVR only improves difficulties the base model already solves with non-zero
  probability, which is why the curriculum starts at one empty cell.
- Results are from single runs on one GPU, not multi-seed benchmarks.

## References

- Shao et al., *DeepSeekMath* (GRPO).
- DeepSeek-AI, *DeepSeek-R1*.
- Unsloth and TRL `GRPOTrainer` documentation.
