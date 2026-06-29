# Experiment exp7_8x8_ceiling

_2026-06-28 21:06_  ·  config `configs/exp7_8x8_ceiling.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `698f6c85`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=5e-06, beta=0.04
- total train time: 56.3 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 8x8 | 10 | 90 | 11% → 18% | 56% | 8.0m |
| 1 | 8x8 | 12 | 130 | 8% → 6% | 75% | 11.3m |
| 2 | 8x8 | 14 | 160 | 6% → 2% | 38% | 15.4m |
| 3 | 8x8 | 16 | 190 | 1% → 0% | 31% | 20.6m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 82% | +75% |
| 4x4 | 2 | 4% | 67% | +63% |
| 4x4 | 3 | 1% | 55% | +54% |
| 4x4 | 4 | 3% | 36% | +33% |
| 4x4 | 5 | 2% | 26% | +24% |
| 4x4 | 6 | 1% | 14% | +13% |
| 4x4 | 7 | 0% | 8% | +8% |
| 4x4 | 8 | 0% | 6% | +6% |
| 4x4 | 9 | 0% | 2% | +2% |
| 4x4 | 10 | 0% | 1% | +1% |
| 6x6 | 2 | 0% | 10% | +10% |
| 6x6 | 4 | 0% | 2% | +2% |
| 6x6 | 6 | 0% | 1% | +1% |
| 6x6 | 8 | 0% | 0% | +0% |
| 6x6 | 10 | 0% | 0% | +0% |
| 6x6 | 12 | 0% | 0% | +0% |
| 6x6 | 15 | 0% | 0% | +0% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 82% | +82% |
| 8x8 | 8 | 0% | 44% | +44% |
| 8x8 | 12 | 0% | 22% | +22% |
| 8x8 | 16 | 0% | 2% | +2% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **None** empty cells, ≥50% up to **3** empty cells.
- **6x6**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **4** empty cells.
