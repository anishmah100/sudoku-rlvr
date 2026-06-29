# Frontier: base vs trained (held-out exact-solve rate, greedy)

100 puzzles per cell, disjoint from training. Trained uses the best adapter
per size (4x4 exp1, 6x6 exp3+exp8, 8x8 exp4-exp7). `b` = base, `t` = trained.

| empty cells | 4x4 b→t | 6x6 b→t | 8x8 b→t |
|---|---|---|---|
| 1 | 7% → 98% | – | 3% → 97% |
| 2 | 4% → 90% | 0% → 91% | 1% → 91% |
| 3 | 1% → 83% | – | 1% → 83% |
| 4 | 3% → 71% | 0% → 85% | 0% → 82% |
| 5 | 2% → 55% | – | – |
| 6 | 1% → 44% | 0% → 63% | – |
| 7 | 0% → 35% | – | – |
| 8 | 0% → 22% | 0% → 45% | 0% → 44% |
| 9 | 0% → 16% | – | – |
| 10 | 0% → 12% | 0% → 25% | – |
| 12 | – | 0% → 15% | 0% → 22% |
| 15 | – | 0% → 4% | – |
| 16 | – | – | 0% → 2% |
| 18 | – | 0% → 3% | – |
| 20 | – | – | 0% → 0% |
| 24 | – | – | 0% → 0% |
| 30 | – | – | 0% → 0% |
