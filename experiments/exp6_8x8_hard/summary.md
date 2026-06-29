# Experiment exp6_8x8_hard

_2026-06-28 20:00_  ·  config `configs/exp6_8x8_hard.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `dc8badc4`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=5e-06, beta=0.04
- total train time: 57.5 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 8x8 | 6 | 90 | 36% → 32% | 94% | 8.3m |
| 1 | 8x8 | 8 | 140 | 24% → 21% | 100% | 14.8m |
| 2 | 8x8 | 10 | 170 | 15% → 9% | 69% | 15.9m |
| 3 | 8x8 | 12 | 190 | 8% → 8% | 50% | 17.6m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 83% | +76% |
| 4x4 | 2 | 4% | 67% | +63% |
| 4x4 | 3 | 1% | 56% | +55% |
| 4x4 | 4 | 3% | 39% | +36% |
| 4x4 | 5 | 2% | 27% | +25% |
| 4x4 | 6 | 1% | 14% | +13% |
| 4x4 | 7 | 0% | 10% | +10% |
| 4x4 | 8 | 0% | 8% | +8% |
| 4x4 | 9 | 0% | 2% | +2% |
| 4x4 | 10 | 0% | 1% | +1% |
| 6x6 | 2 | 0% | 10% | +10% |
| 6x6 | 4 | 0% | 4% | +4% |
| 6x6 | 6 | 0% | 1% | +1% |
| 6x6 | 8 | 0% | 0% | +0% |
| 6x6 | 10 | 0% | 0% | +0% |
| 6x6 | 12 | 0% | 0% | +0% |
| 6x6 | 15 | 0% | 0% | +0% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 78% | +78% |
| 8x8 | 8 | 0% | 40% | +40% |
| 8x8 | 12 | 0% | 16% | +16% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 2% | +2% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **None** empty cells, ≥50% up to **3** empty cells.
- **6x6**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **4** empty cells.
