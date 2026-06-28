# Before vs After — solve rate by difficulty

Exact-solve rate on 100 fresh puzzles per cell, greedy decoding. The 4×4
curriculum was the only thing trained; 6×6/9×9 rows show zero-shot transfer.

| Puzzle | Empty cells | Before | After | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 99% | +92% |
| 4x4 | 2 | 0% | 93% | +93% |
| 4x4 | 3 | 3% | 86% | +83% |
| 4x4 | 4 | 0% | 73% | +73% |
| 4x4 | 6 | 0% | 46% | +46% |
| 4x4 | 8 | 0% | 32% | +32% |
| 6x6 | 2 | 0% | 17% | +17% |
| 6x6 | 4 | 0% | 4% | +4% |
| 6x6 | 8 | 0% | 0% | +0% |
| 6x6 | 12 | 0% | 0% | +0% |
| 9x9 | 4 | 4% | 19% | +15% |
| 9x9 | 8 | 0% | 3% | +3% |
| 9x9 | 16 | 0% | 0% | +0% |
