# Methodology

A detailed account of the task, reward, curriculum, algorithm, and evaluation
protocol, plus threats to validity. The intent is that a reader can judge whether
the headline claim — *a 3B specialist beats general frontier models on Sudoku* — is
earned, and reproduce it.

## 1. Task formalization

An `n×n` Sudoku with box dimensions `(b_r, b_c)` where `b_r · b_c = n`:

| n | box | digits |
|---|-----|--------|
| 4 | 2×2 | 1–4 |
| 6 | 2×3 | 1–6 |
| 9 | 3×3 | 1–9 |

A **puzzle** is a partial assignment (the *givens*/clues); a **solution** is a full
assignment satisfying the row/column/box all-different constraints. Generation
(`sudoku.py`) draws a random complete grid by randomized backtracking, then removes
cells in random order, keeping a removal **only if the puzzle still has exactly one
solution** (checked by an MRV-ordered solver counting up to two solutions). This
guarantees the reward target is well-defined: there is a single correct grid.

Clue counts are chosen to sit in a learnable band for short-CoT RL: 8 (4×4), 18
(6×6), 40 (9×9).

## 2. Reward function

Let `G` be the parsed grid, `P` the puzzle, `S` the unique solution, and `E` the set
of empty cells in `P`. Define:

```
format_ok  = (G is a clean n×n block of digits 1..n extracted from <answer>…</answer>)
givens_ok  = ∀ (r,c) with P[r,c] ≠ 0 :  G[r,c] = P[r,c]
frac       = |{ (r,c) ∈ E : G[r,c] = S[r,c] }| / |E|
solved     = givens_ok ∧ (G = S)
```

The scalar reward:

```
r = 0                              if not format_ok
  = 0.20                           (format)
    − 0.50·[¬givens_ok]            (penalty for overwriting a clue)
    + 1.00·frac                    (dense cell accuracy)
    + 2.00·[solved]                (exact-solve bonus)
```

**Why dense?** GRPO computes each sample's advantage by standardizing rewards *within
its group* of `G` completions for the same prompt. A purely binary "solved" reward is
near-constant (all zeros) early in training → zero advantage → no gradient. The
`frac` term varies across completions immediately, so the group always has a spread to
learn from. The exact-solve bonus then creates a sharp incentive to convert
"mostly right" into "exactly right," countering reward-hacking toward partial credit.

The function is implemented once (`rewards.score`) and reused by both the trainer and
the evaluator, so training signal and reported metrics are by-construction consistent.

## 3. Curriculum

Two curricula are provided. **The size curriculum** (`configs/curriculum.yaml`:
4×4→6×6→9×9 at normal clue counts) is the intuitive design — and it **failed** (see
[RESULTS.md](RESULTS.md) §2): a 3B model never solves a 6×6/9×9 cold, so the
solve-bonus signal never fires and GRPO has nothing to amplify.

**The difficulty curriculum** (`configs/easy_curriculum.yaml`) is what worked. It stays
on 4×4 and ramps the number of *empty cells* — starting where the base model can
already occasionally succeed:

| Stage | Board | Empty cells | Clues | Steps | Rationale |
|---|---|---|---|---|---|
| 0 | 4×4 | 1 | 15 | 120 | base ≈9% here — a real bootstrap |
| 1 | 4×4 | 2 | 14 | 150 | add one degree of freedom |
| 2 | 4×4 | 3 | 13 | 150 | |
| 3 | 4×4 | 5 | 11 | 200 | |
| 4 | 4×4 | 7 | 9 | 250 | full-ish 4×4 |

The governing principle is the **bootstrap requirement**: RLVR *amplifies* behaviors
the policy already exhibits with non-zero probability; it does not *create* them. The
curriculum must therefore begin at a difficulty where the base success rate is
measurably above zero (quantified by `scripts/difficulty_sweep.py`). Each stage's data
is generated in-process at the configured clue count.

## 4. Algorithm: GRPO

Group Relative Policy Optimization (Shao et al., DeepSeekMath; popularized by
DeepSeek-R1):

1. For prompt `q`, sample a group of `K` completions `{o_1..o_K}` (here `K = 8`).
2. Score each with the verifiable reward `r_i`.
3. Advantage `A_i = (r_i − mean(r)) / std(r)` — purely *relative* within the group.
4. Policy-gradient update toward high-advantage completions, with a KL penalty to the
   reference policy.

Crucially there is **no learned value/critic network** (unlike PPO), which roughly
halves memory — the reason this fits in 24 GB alongside a vLLM generation engine.

### Hyperparameters (`configs/curriculum.yaml`)

| | value |
|---|---|
| backbone | Qwen2.5-3B-Instruct, 4-bit |
| LoRA rank / α | 32 / 32, targets = all attention + MLP projections |
| group size `K` | 8 |
| batch (prompts/step) | 8 |
| learning rate | 5e-6, cosine, 10% warmup |
| optimizer | adamw_8bit, weight decay 0.1 |
| sampling (train) | temperature 1.0 |
| max prompt / completion | 1024 / {512,768,1280} |
| gpu_memory_utilization | 0.6 (rest reserved for training) |

## 5. Evaluation protocol

- **Held-out sets**: 200 puzzles per size, generated with a *different seed* from
  training (`seed + 10000`), so no puzzle overlap.
- **Decoding**: greedy (temperature 0) for determinism.
- **Metrics** (`evaluate.py`):
  - `solve_rate` — fraction exactly solved (the headline number);
  - `frac_correct` — mean per-cell accuracy on empties;
  - `format_rate` — fraction emitting a parseable grid;
  - `givens_ok_rate` — fraction preserving all clues;
  - `mean_reward`.
- **Before vs after**: identical eval is run on the base model (`--adapter none`) and
  the trained adapter (`--adapter outputs/adapter`); `plot.py` renders the comparison.

### Frontier reference
To contextualize "beats frontier," the same held-out puzzles and the same parser can
be sent to a general frontier chat model (no tools); its `solve_rate` is the bar the
specialist is measured against. This is the apples-to-apples comparison the thesis
rests on. (Optional add-on; not required to run the core study.)

## 6. Threats to validity

- **Single run, single seed.** Reported as a case study; multi-seed CIs would
  strengthen it.
- **Difficulty.** 40-clue 9×9 puzzles lean toward propagation-solvable; harder bands
  (fewer clues, forced backtracking) are a distinct, harder claim.
- **Reward hacking.** The dense term could in principle be gamed by filling
  high-probability cells; the exact-solve bonus and the `givens_ok` penalty are the
  guards, and `solve_rate` (not `frac_correct`) is treated as the headline.
- **Contamination.** Puzzles are randomly generated at run time, so the eval set
  cannot have been in any pretraining corpus.
- **Format sensitivity.** A weak base format rate can understate a frontier model's
  true ability; the parser is deliberately tolerant (`.`/`0`/`_` blanks, commas,
  extra whitespace) to avoid penalizing reasonable formats.
