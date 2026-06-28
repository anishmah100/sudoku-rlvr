#!/usr/bin/env python3
"""A/B: does direct-answer (no CoT) beat short-CoT for the BASE model on easy 4x4?"""
import os
import sys

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sudoku_rlvr.data import build_examples  # noqa: E402
from sudoku_rlvr.modeling import generate, load_model  # noqa: E402
from sudoku_rlvr.rewards import score  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/curriculum.yaml")))
model, tok = load_model(cfg["model"], cfg["max_seq_length"], cfg["lora_rank"],
                        cfg["gpu_memory_utilization"], True, for_training=True)

for n, empties in [(4, 1), (4, 2), (4, 4), (6, 4), (6, 8)]:
    clues = n * n - empties
    for cot in (True, False):
        exs = build_examples(n, 100, seed=777, clues=clues, cot=cot)
        maxtok = 600 if cot else 120
        texts = generate(model, tok, [e["prompt"] for e in exs], maxtok, temperature=0.0)
        sc = [score(t, e["puzzle"], e["solution"], n) for t, e in zip(texts, exs)]
        solve = sum(s["solved"] for s in sc) / len(sc)
        fmt = sum(s["format_ok"] for s in sc) / len(sc)
        frac = sum(s["frac_correct"] for s in sc) / len(sc)
        print(f"{n}x{n} empties={empties} {'CoT   ' if cot else 'DIRECT'}: "
              f"solve={solve:5.1%} fmt={fmt:5.1%} frac={frac:5.1%}")
