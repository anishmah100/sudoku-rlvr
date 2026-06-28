# Experiment exp5_8x8_mid

_2026-06-28 18:54_  ·  config `configs/exp5_8x8_mid.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `a8177e29`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=5e-06, beta=0.04
- total train time: 59.0 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 8x8 | 2 | 60 | 32% → 39% | 81% | 6.9m |
| 1 | 8x8 | 3 | 90 | 22% → 22% | 88% | 8.5m |
| 2 | 8x8 | 4 | 120 | 15% → 14% | 56% | 11.6m |
| 3 | 8x8 | 6 | 150 | 6% → 18% | 69% | 15.0m |
| 4 | 8x8 | 8 | 170 | 2% → 14% | 56% | 16.1m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 78% | +71% |
| 4x4 | 2 | 4% | 52% | +48% |
| 4x4 | 3 | 1% | 47% | +46% |
| 4x4 | 4 | 3% | 29% | +26% |
| 4x4 | 5 | 2% | 17% | +15% |
| 4x4 | 6 | 1% | 6% | +5% |
| 4x4 | 7 | 0% | 8% | +8% |
| 4x4 | 8 | 0% | 4% | +4% |
| 4x4 | 9 | 0% | 2% | +2% |
| 4x4 | 10 | 0% | 1% | +1% |
| 6x6 | 2 | 0% | 39% | +39% |
| 6x6 | 4 | 0% | 14% | +14% |
| 6x6 | 6 | 0% | 2% | +2% |
| 6x6 | 8 | 0% | 1% | +1% |
| 6x6 | 10 | 0% | 0% | +0% |
| 6x6 | 12 | 0% | 0% | +0% |
| 6x6 | 15 | 0% | 0% | +0% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 61% | +61% |
| 8x8 | 8 | 0% | 20% | +20% |
| 8x8 | 12 | 0% | 7% | +7% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **None** empty cells, ≥50% up to **2** empty cells.
- **6x6**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **4** empty cells.
