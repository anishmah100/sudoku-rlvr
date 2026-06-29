# Experiment exp8_6x6_hard

_2026-06-28 22:16_  ·  config `configs/exp8_6x6_hard.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `3215130c`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=5e-06, beta=0.04
- total train time: 58.2 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 6x6 | 8 | 90 | 27% → 36% | 94% | 10.6m |
| 1 | 6x6 | 10 | 130 | 16% → 6% | 69% | 11.9m |
| 2 | 6x6 | 12 | 160 | 6% → 21% | 81% | 18.8m |
| 3 | 6x6 | 15 | 180 | 3% → 1% | 19% | 16.1m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 99% | +92% |
| 4x4 | 2 | 4% | 94% | +90% |
| 4x4 | 3 | 1% | 90% | +89% |
| 4x4 | 4 | 3% | 81% | +78% |
| 4x4 | 5 | 2% | 62% | +60% |
| 4x4 | 6 | 1% | 52% | +51% |
| 4x4 | 7 | 0% | 38% | +38% |
| 4x4 | 8 | 0% | 37% | +37% |
| 4x4 | 9 | 0% | 20% | +20% |
| 4x4 | 10 | 0% | 13% | +13% |
| 6x6 | 2 | 0% | 91% | +91% |
| 6x6 | 4 | 0% | 85% | +85% |
| 6x6 | 6 | 0% | 63% | +63% |
| 6x6 | 8 | 0% | 45% | +45% |
| 6x6 | 10 | 0% | 25% | +25% |
| 6x6 | 12 | 0% | 15% | +15% |
| 6x6 | 15 | 0% | 4% | +4% |
| 6x6 | 18 | 0% | 3% | +3% |
| 8x8 | 4 | 0% | 7% | +7% |
| 8x8 | 8 | 0% | 2% | +2% |
| 8x8 | 12 | 0% | 0% | +0% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **3** empty cells, ≥50% up to **6** empty cells.
- **6x6**: solved ≥90% up to **2** empty cells, ≥50% up to **6** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
