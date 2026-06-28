#!/usr/bin/env python3
"""Evaluate a model (base or LoRA adapter) on the held-out eval sets.

Writes results/eval_<tag>.json with per-size and overall metrics:
  solve_rate, frac_correct, format_rate, givens_ok_rate, mean_reward.

    # baseline (no adapter)
    python scripts/evaluate.py --adapter none --tag base
    # trained
    python scripts/evaluate.py --adapter outputs/adapter --tag trained
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sudoku_rlvr.data import read_jsonl  # noqa: E402
from sudoku_rlvr.modeling import generate, load_model  # noqa: E402
from sudoku_rlvr.rewards import score  # noqa: E402

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def aggregate(scores):
    k = len(scores)
    return {
        "n": k,
        "solve_rate": sum(s["solved"] for s in scores) / k,
        "frac_correct": sum(s["frac_correct"] for s in scores) / k,
        "format_rate": sum(s["format_ok"] for s in scores) / k,
        "givens_ok_rate": sum(s["givens_ok"] for s in scores) / k,
        "mean_reward": sum(s["reward"] for s in scores) / k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/curriculum.yaml"))
    ap.add_argument("--adapter", default="none", help="'none' for base, else adapter dir")
    ap.add_argument("--tag", required=True, help="label for output file, e.g. base/trained")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 9])
    ap.add_argument("--limit", type=int, default=None, help="cap puzzles per size")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ev = cfg["eval"]

    model, tokenizer = load_model(
        model_name=cfg["model"],
        max_seq_length=cfg["max_seq_length"],
        lora_rank=cfg["lora_rank"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        # Attach LoRA scaffolding so `model.load_lora` is available; freshly
        # initialized LoRA is a no-op (B=0), so base eval (lora_path=None) still
        # reflects the true base model.
        for_training=True,
    )
    lora_path = None if args.adapter == "none" else args.adapter

    os.makedirs(RESULTS, exist_ok=True)
    report = {"tag": args.tag, "adapter": args.adapter, "by_size": {}, "samples": {}}
    all_scores = []
    for n in args.sizes:
        rows = read_jsonl(os.path.join(DATA, f"eval_{n}x{n}.jsonl"))
        if args.limit:
            rows = rows[: args.limit]
        chats = [r["prompt"] for r in rows]
        texts = generate(model, tokenizer, chats, ev["max_new_tokens"],
                         temperature=ev["temperature"], lora_path=lora_path)
        scores = [score(t, r["puzzle"], r["solution"], n)
                  for t, r in zip(texts, rows)]
        all_scores += scores
        report["by_size"][str(n)] = aggregate(scores)
        report["samples"][str(n)] = {"prompt": chats[0][-1]["content"],
                                     "completion": texts[0]}
        m = report["by_size"][str(n)]
        print(f"[{args.tag}] {n}x{n}: solve={m['solve_rate']:.1%} "
              f"frac={m['frac_correct']:.1%} fmt={m['format_rate']:.1%} "
              f"reward={m['mean_reward']:.3f}")
    report["overall"] = aggregate(all_scores)

    out = os.path.join(RESULTS, f"eval_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
