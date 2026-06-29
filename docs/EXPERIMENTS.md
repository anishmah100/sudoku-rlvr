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

The 4×4 adapter already solves 6×6 at 2 empty (29%) and 4 empty (17%) by transfer, so
this resumes from it. 6×6 difficulty curriculum from 2 empty upward, K=8,
gradient_accumulation 2, LR 5e-6, max_grad_norm 0.5.

Held-out solve rate after training, 6×6 by empty cells: 2→90, 4→84, 6→58, 8→41, 10→24,
12→11, 15→4, 18→0 (%); format recovered to 95–100% (the 4×4 adapter alone produced
malformed 6×6 grids ~54% of the time). Base was 0% at every 6×6 difficulty. Adapter
saved as `models/exp3_6x6`.

## exp6_8x8_hard — push 6–12 empty (`configs/exp6_8x8_hard.yaml`)

Resumes exp5 and concentrates steps on 6 → 12 empty. Held-out 8×8 after training: 1→96%,
2→86%, 3→76%, 4→78%, 8→40%, 12→16%, 16→0%. Format is 100% at every difficulty including
the unsolved hard end. Training the hard-mid range also raised the easy end (1 empty
90%→96%). The minimal-clue end (≥16 empty) stays at 0%. Adapter saved as `models/exp6_8x8`.

## exp7_8x8_ceiling — hard 8×8 with longer reasoning (`configs/exp7_8x8_ceiling.yaml`)

Resumes exp6 and trains 10 → 16 empty with a larger completion budget. Held-out 8×8
after training: 1→97%, 2→91%, 3→83%, 4→82%, 8→44%, 12→22%, 16→2%, 20+→0%. Gains over
exp6 are small (e12 16%→22%, e16 0%→2%) compared with the large jumps in exp4–exp6, so
the 8×8 frontier has roughly converged for this model and training scale: solvable to
about 12 empty cells at declining rates, with a wall near 16. A larger reasoning budget
did not move the wall, so the limit is the model's search ability, not output length.
Adapter saved as `models/exp7_8x8`.

## exp8_6x6_hard — push hard 6×6 (`configs/exp8_6x6_hard.yaml`)

Resumes exp3 and trains 8 → 15 empty. Held-out 6×6 after training: 2→91%, 4→85%, 6→63%,
8→45%, 10→25%, 12→15%, 15→4%, 18→3%. Gains over exp3 are small (e8 41%→45%, e12 11%→15%),
so 6×6 was already near its converged frontier after the single exp3 run — unlike 8×8,
which needed several runs because it started from the transcription wall. Adapter saved
as `models/exp8_6x6`.

## Achievable frontier (4×4, 6×6, 8×8)

Held-out exact-solve rate by board size and number of empty cells (8×8 after the
transcription warm-up, exp4):

| empty cells | 4×4 | 6×6 | 8×8 |
|---|---|---|---|
| 1 | 98% | 91%¹ | 97% |
| 2 | 90% | 91% | 91% |
| 3–4 | 71–83% | 85% | 82–83% |
| 8 | 22% | 45% | 44% |
| 10–12 | 12% | 25–15% | 22% |
| 15–16 | — | 4% | 2% |
| 18+ | — | 3→0% | 0% |

¹ 6×6 column measured from 2 empty up; 8×8 column after the transcription warm-up plus
mid-range training (exp4, exp5). All sizes are solved at the easy end and decline toward
the minimal-clue end. The reachable difficulty shrinks as the board grows, and 8×8
required an explicit transcription warm-up (a 0-empty copy task) before any solving,
because reproducing the 64-cell grid is the prerequisite the base model fails. The hard
(minimal-clue) end of every size stays at 0%, consistent with the model not performing
reliable backtracking search through its reasoning. Resuming and training further keeps
lifting the frontier (e.g. 8×8 at 4 empty went 9% → 61% with an additional run), so
these numbers are a function of training budget, not a hard limit — except the
minimal-clue end, which does not move.

## exp4_8x8_transcribe — transcription warm-up (`configs/exp4_8x8_transcribe.yaml`)

8×8 from base with a stage-0 pure copy task (0 empty cells) to train faithful 64-cell
reproduction, then a difficulty ramp 1 → 4 empty.

The copy stage reaches 100% exact reproduction during training. Held-out solve rate
after training, 8×8 by empty cells: 1→61%, 2→39%, 3→22%, 4→9%, 8→2%, 12+→0%. Format
rose to 94–99% at the easy end (base ~50%, and exp2 had collapsed it to ~6%), and the
fraction of completions preserving all givens rose from 3% (base) to 80% at 1 empty.

The transcription warm-up removes the format/transcription bottleneck, which makes easy
8×8 solvable. The hard 8×8 end stays near 0%, as with the other sizes. Adapter saved as
`models/exp4_8x8`.

## exp5_8x8_mid — push 8×8 difficulty (`configs/exp5_8x8_mid.yaml`)

Resumes exp4 and ramps 2 → 8 empty with more steps. Held-out 8×8 after training: 1→90%,
2→77%, 3→68%, 4→61%, 8→20%, 12→7%, 16→0%. Continued training improved both the easy end
(1 empty 61%→90%, givens preserved now 100%) and the mid range (4 empty 9%→61%, 8 empty
2%→20%). The frontier extends with more training but the minimal-clue end (≥16 empty)
stays at 0%. Adapter saved as `models/exp5_8x8`.
