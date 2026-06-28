# Experiment exp4_8x8_transcribe

_2026-06-28 17:45_  ·  config `configs/exp4_8x8_transcribe.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `76cd3ad0`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=1e-05, beta=0.04
- total train time: 95.2 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 8x8 | 0 | 80 | 20% → 100% | 100% | 7.9m |
| 1 | 8x8 | 1 | 120 | 6% → 21% | 88% | 17.2m |
| 2 | 8x8 | 2 | 140 | 6% → 22% | 81% | 20.6m |
| 3 | 8x8 | 3 | 160 | 2% → 5% | 56% | 30.1m |
| 4 | 8x8 | 4 | 160 | 1% → 0% | 38% | 18.6m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 63% | +56% |
| 4x4 | 2 | 4% | 34% | +30% |
| 4x4 | 3 | 1% | 27% | +26% |
| 4x4 | 4 | 3% | 20% | +17% |
| 4x4 | 5 | 2% | 6% | +4% |
| 4x4 | 6 | 1% | 4% | +3% |
| 4x4 | 7 | 0% | 6% | +6% |
| 4x4 | 8 | 0% | 3% | +3% |
| 4x4 | 9 | 0% | 3% | +3% |
| 4x4 | 10 | 0% | 2% | +2% |
| 6x6 | 2 | 0% | 36% | +36% |
| 6x6 | 4 | 0% | 13% | +13% |
| 6x6 | 6 | 0% | 3% | +3% |
| 6x6 | 8 | 0% | 2% | +2% |
| 6x6 | 10 | 0% | 1% | +1% |
| 6x6 | 12 | 0% | 1% | +1% |
| 6x6 | 15 | 0% | 0% | +0% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 9% | +9% |
| 8x8 | 8 | 0% | 2% | +2% |
| 8x8 | 12 | 0% | 0% | +0% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **None** empty cells, ≥50% up to **1** empty cells.
- **6x6**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
