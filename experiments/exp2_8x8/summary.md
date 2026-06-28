# Experiment exp2_8x8

_2026-06-28 14:28_  ·  config `configs/exp2_8x8.yaml`  ·  GPU NVIDIA GeForce RTX 4090  ·  git `6ff88523`

## Config

- model: `Qwen/Qwen2.5-3B-Instruct`, LoRA rank 32, cot=True
- GRPO: K=8, batch=8, grad_accum=2, LR=1e-05, beta=0.04
- total train time: 118.0 min

## Training (per stage)

| stage | size | empties | steps | solved first10 → last10 | max | min |
|---|---|---|---|---|---|---|
| 0 | 8x8 | 1 | 60 | 10% → 15% | 44% | 12.5m |
| 1 | 8x8 | 2 | 70 | 1% → 6% | 31% | 15.4m |
| 2 | 8x8 | 3 | 80 | 1% → 1% | 12% | 17.8m |
| 3 | 8x8 | 4 | 100 | 0% → 0% | 12% | 17.4m |
| 4 | 8x8 | 6 | 130 | 0% → 1% | 6% | 22.2m |
| 5 | 8x8 | 8 | 160 | 0% → 0% | 0% | 31.8m |

## Before vs after (solve rate by difficulty)

| puzzle | empty | before | after | Δ |
|---|---|---|---|---|
| 4x4 | 1 | 7% | 22% | +15% |
| 4x4 | 2 | 4% | 11% | +7% |
| 4x4 | 3 | 1% | 5% | +4% |
| 4x4 | 4 | 3% | 3% | +0% |
| 4x4 | 5 | 2% | 3% | +1% |
| 4x4 | 6 | 1% | 0% | -1% |
| 4x4 | 7 | 0% | 2% | +2% |
| 4x4 | 8 | 0% | 0% | +0% |
| 4x4 | 9 | 0% | 0% | +0% |
| 4x4 | 10 | 0% | 0% | +0% |
| 6x6 | 2 | 0% | 8% | +8% |
| 6x6 | 4 | 0% | 1% | +1% |
| 6x6 | 6 | 0% | 0% | +0% |
| 6x6 | 8 | 0% | 0% | +0% |
| 6x6 | 10 | 0% | 0% | +0% |
| 6x6 | 12 | 0% | 0% | +0% |
| 6x6 | 15 | 0% | 0% | +0% |
| 6x6 | 18 | 0% | 0% | +0% |
| 8x8 | 4 | 0% | 0% | +0% |
| 8x8 | 8 | 0% | 0% | +0% |
| 8x8 | 12 | 0% | 0% | +0% |
| 8x8 | 16 | 0% | 0% | +0% |
| 8x8 | 20 | 0% | 0% | +0% |
| 8x8 | 24 | 0% | 0% | +0% |
| 8x8 | 30 | 0% | 0% | +0% |

## Summary

- **4x4**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **6x6**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
- **8x8**: solved ≥90% up to **None** empty cells, ≥50% up to **None** empty cells.
