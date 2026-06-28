# Observations

Auto-generated from the difficulty sweep (`results/difficulty_sweep_*.json`).

- **4×4 mastery:** exact-solve rate on the trivial case (1 empty) rose 7% → 99%; on full 4×4 (8 empties) 0% → 32%.
- **Graceful difficulty falloff:** after training, solve rate decays smoothly with empties — 1→99%, 2→93%, 3→86%, 4→73%, 6→46%, 8→32%.
- **Zero-shot transfer to larger boards (never trained):** 6x6/2-empty 0%→17%, 6x6/4-empty 0%→4%, 9x9/4-empty 4%→19%, 9x9/8-empty 0%→3%.
- **Baseline reminder:** the untrained model solves ~0% of all but the most trivial instances, so essentially the entire after-curve is new capability.

> Scope: single RTX 4090, single seed, Qwen2.5-3B + LoRA, CoT decoding. The curriculum trains 4×4 only; larger boards are out-of-distribution.
