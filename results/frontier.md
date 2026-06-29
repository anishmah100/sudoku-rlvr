# Frontier (held-out exact-solve rate, greedy)

Base model is ~0% at every point. Trained columns use the best adapter per
size (4x4 exp1, 6x6 exp3+exp8, 8x8 exp4-exp7).

| empty cells | 4x4 | 6x6 | 8x8 |
|---|---|---|---|
| 1 | 98% | – | 97% |
| 2 | 90% | 91% | 91% |
| 3 | 83% | – | 83% |
| 4 | 71% | 85% | 82% |
| 5 | 55% | – | – |
| 6 | 44% | 63% | – |
| 7 | 35% | – | – |
| 8 | 22% | 45% | 44% |
| 9 | 16% | – | – |
| 10 | 12% | 25% | – |
| 12 | – | 15% | 22% |
| 15 | – | 4% | – |
| 16 | – | – | 2% |
| 18 | – | 3% | – |
| 20 | – | – | 0% |
| 24 | – | – | 0% |
| 30 | – | – | 0% |
