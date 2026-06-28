# Experiment exp1_4x4

_2026-06-28 00:59_  ·  config `configs/exp1_4x4.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `bbebe53b`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=4, LR=5e-06, beta=0.04
- total train time: 63.7 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 4x4 | 5 | 20 | 70% → 56% | 97% | 3.3m |
| 1 | 4x4 | 6 | 30 | 35% → 45% | 69% | 5.1m |
| 2 | 4x4 | 7 | 45 | 36% → 34% | 69% | 8.1m |
| 3 | 4x4 | 8 | 65 | 19% → 22% | 59% | 9.8m |
| 4 | 4x4 | 9 | 90 | 23% → 13% | 50% | 13.1m |
| 5 | 4x4 | 10 | 120 | 5% → 8% | 31% | 23.5m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 98% | +91% |
| 4x4 | 2 | 4% | 90% | +86% |
| 4x4 | 3 | 1% | 83% | +82% |
| 4x4 | 4 | 3% | 71% | +68% |
| 4x4 | 5 | 2% | 55% | +53% |
| 4x4 | 6 | 1% | 44% | +43% |
| 4x4 | 7 | 0% | 35% | +35% |
| 4x4 | 8 | 0% | 22% | +22% |
| 4x4 | 9 | 0% | 16% | +16% |
| 4x4 | 10 | 0% | 12% | +12% |
| 6x6 | 2 | 0% | 29% | +29% |
| 6x6 | 4 | 0% | 17% | +17% |
| 6x6 | 6 | 0% | 7% | +7% |
| 6x6 | 8 | 0% | 2% | +2% |
| 6x6 | 10 | 0% | 0% | +0% |
| 6x6 | 12 | 0% | 0% | +0% |
| 6x6 | 15 | 0% | 0% | +0% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 2% | +2% |
| 8x8 | 8 | 0% | 0% | +0% |
| 8x8 | 12 | 0% | 0% | +0% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **2** empty cells, ≥50% up to **5** empty cells.
- **6x6**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
