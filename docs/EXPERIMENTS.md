# Experiment log

A chronological record of the runs, the configuration each used, the measured
result, and the decision it led to. All solve rates are exact-solve on held-out
puzzles (100 per difficulty cell, greedy decoding) unless noted. Per-run artifacts
live in `experiments/<id>/` (config, `meta.json`, per-stage `train_log_stage*.json`,
`sweep.json`, charts, `summary.md`); the table in `experiments/REGISTRY.md` lists each
run.

## Harness and controls

- **Reward** (`rewards.score`): +0.20 parseable grid, −0.50 if a given clue is
  overwritten, +1.00 × fraction of empty cells correct, +2.00 if the grid exactly
  matches the unique solution. Used by both training and evaluation.
- **Train/eval split**: each puzzle is assigned to a train or eval pool by a hash of
  its string (`data.py`, 15% held out), so the sets are disjoint independent of seed.
  This matters for small spaces (e.g. 4×4 with one empty cell has only a few thousand
  puzzles).
- **Tests** (`tests/test_harness.py`): ground-truth solutions score as solved at every
  size/difficulty; corrupted grids never score as solved; train and eval splits are
  disjoint; generated puzzles are valid and unique.

## Baseline

Base Qwen2.5-3B (no adapter), held-out sweep (`results/difficulty_sweep_base.json`):

- 4×4: solve rate 7% at 1 empty, ~0–3% at 2–6 empty, 0% from 7 empty up.
- 6×6: 0% at all measured difficulties; format 81–92%.
- 8×8: 0% solve from 4 empty up; format ~40–60%. At 1 empty: 3% solve, 81% format, but
  only 3% of completions preserve all 63 given cells.

The base model rarely produces a correct grid and frequently corrupts given cells while
transcribing. Reading completions shows its reasoning is often internally inconsistent
(states a constraint, then violates it).

## Size-based curriculum (negative result)

`configs/curriculum.yaml`: 4×4 → 6×6 → 9×9 at standard clue counts, LR 5e-6.

Result: no improvement (4×4 0→1%, 6×6/9×9 0→0%); format on 6×6/9×9 dropped. Training
logs show KL to the reference at 0.001–0.007 (policy barely moved) and `solved` = 0 for
every step on 6×6/9×9. With no correct completions, GRPO has no signal on those sizes.

Decision: the curriculum must begin where the base model already solves some puzzles.

## Direct vs reasoning prompt

A/B on 4×4 at 1 empty: with reasoning, 9% solve / 90% format; direct (grid only), 0%
solve / 100% format. The model only fills cells when it reasons first, so the reasoning
prompt is kept (`prompts.py` retains a direct mode for reference).

## Difficulty curriculum, 4×4 (`easy_curriculum.yaml`)

4×4 only, increasing empty cells per stage (1 → 7), LoRA carried forward, from base,
K=8, LR 1e-5. Adapter saved as `models/v1_4x4_easy`.

Held-out solve rate after training: 1 empty 99%, 2 empty 93%, 3 empty 86%, 4 empty 73%,
6 empty 46%, 8 empty 32%. Per-stage `solved` climbs within each stage. This is the first
configuration that produced a clear before/after.

## exp1_4x4 — hard 4×4 (`configs/exp1_4x4.yaml`)

Goal: push 4×4 to minimal-clue boards. Resumes `models/v1_4x4_easy` and ramps 5 → 10
empty cells.

First attempt (LR 1e-5, gradient_accumulation 1) reached 100% on 5 empty within ~25
steps, then a KL spike (~0.94) destabilized it and solve rate collapsed. Stabilized with
gradient_accumulation 4, LR 5e-6, and max_grad_norm 0.5; KL then stayed ≤ ~0.4.

Held-out solve rate after training, 4×4 by empty cells: 1→98, 2→90, 3→83, 4→71, 5→55,
6→44, 7→35, 8→22, 9→16, 10→12 (%). The frontier extends (9–10 empty become nonzero) but
the minimal-clue end stays ~12–22%. Minimal-clue 4×4 requires backtracking/search, which
this model does not perform reliably through chain-of-thought; this is a soft ceiling.

## Transfer and format observations

Evaluating the exp1_4x4 adapter (trained only on 4×4) on other sizes:

- 6×6: solve rate rises from 0% to 29% at 2 empty, 17% at 4 empty — transfer of
  constraint-following.
- 8×8: format rate drops from ~47% (base) to 3%. The adapter overfits to the 4×4 grid
  shape and emits malformed 8×8 grids.

Decision: train 8×8 from the base model, not from a 4×4 adapter.

## exp2_8x8 — 8×8 from base (`configs/exp2_8x8.yaml`)

8×8 difficulty curriculum from base, 1 → 8 empty cells, K=8, gradient_accumulation 2,
LR 1e-5, max_grad_norm 0.5.

Training `solved` by stage: 1 empty 10→15% (peak 44%), 2 empty 1→6%, 3 empty ~1%,
4–8 empty ~0%. Held-out solve rate stayed 0% at 4 empty and above, and the greedy
format rate fell from ~50% (base) to ~3–10%.

The model cannot reliably transcribe a 64-cell grid, which is a prerequisite to
solving; only the 1-empty case shows a weak signal. The later stages never solve, so
they provide no positive reward, and the policy drifts toward malformed output, which
lowers the format rate. 8×8 is beyond this model at this scale. The reachable rung
above 4×4 is 6×6, pursued next.

## exp3_6x6 — 6×6 from the 4×4 adapter (`configs/exp3_6x6.yaml`)

In progress. The 4×4 adapter already solves 6×6 at 2 empty (29%) and 4 empty (17%) by
transfer, so this resumes from it rather than starting at 0%. 6×6 difficulty curriculum
from 2 empty upward. Results recorded in `experiments/exp3_6x6/` on completion.
