# Experiment exp3_6x6

_2026-06-28 15:59_  ·  config `configs/exp3_6x6.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `776928f2`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=5e-06, beta=0.04
- total train time: 85.8 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 6x6 | 2 | 50 | 38% → 44% | 94% | 9.3m |
| 1 | 6x6 | 3 | 60 | 39% → 54% | 100% | 10.2m |
| 2 | 6x6 | 4 | 80 | 38% → 38% | 100% | 12.9m |
| 3 | 6x6 | 6 | 110 | 31% → 38% | 88% | 15.9m |
| 4 | 6x6 | 8 | 140 | 38% → 34% | 88% | 18.5m |
| 5 | 6x6 | 10 | 160 | 9% → 10% | 75% | 18.3m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 98% | +91% |
| 4x4 | 2 | 4% | 92% | +88% |
| 4x4 | 3 | 1% | 87% | +86% |
| 4x4 | 4 | 3% | 76% | +73% |
| 4x4 | 5 | 2% | 55% | +53% |
| 4x4 | 6 | 1% | 50% | +49% |
| 4x4 | 7 | 0% | 33% | +33% |
| 4x4 | 8 | 0% | 32% | +32% |
| 4x4 | 9 | 0% | 20% | +20% |
| 4x4 | 10 | 0% | 9% | +9% |
| 6x6 | 2 | 0% | 90% | +90% |
| 6x6 | 4 | 0% | 84% | +84% |
| 6x6 | 6 | 0% | 58% | +58% |
| 6x6 | 8 | 0% | 41% | +41% |
| 6x6 | 10 | 0% | 24% | +24% |
| 6x6 | 12 | 0% | 11% | +11% |
| 6x6 | 15 | 0% | 4% | +4% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 8% | +8% |
| 8x8 | 8 | 0% | 1% | +1% |
| 8x8 | 12 | 0% | 0% | +0% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **2** empty cells, ≥50% up to **6** empty cells.
- **6x6**: solved ≥90% up to **2** empty cells, ≥50% up to **6** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
