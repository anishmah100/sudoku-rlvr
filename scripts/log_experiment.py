#!/usr/bin/env python3
"""Summarize one experiment and append it to experiments/REGISTRY.md.

Reads <exp>/meta.json (training) + <exp>/sweep.json (after) + a baseline sweep
(before), computes before/after deltas, writes <exp>/summary.md, and appends a
one-row entry to the registry for cross-experiment study.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXPROOT = os.path.join(ROOT, "experiments")
REGISTRY = os.path.join(EXPROOT, "REGISTRY.md")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def by_key(sweep):
    return {(r["size"], r["empties"]): r for r in sweep} if sweep else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--base", default=os.path.join(ROOT, "results/difficulty_sweep_base.json"))
    args = ap.parse_args()
    exp = os.path.abspath(args.exp_dir)
    expid = os.path.basename(exp.rstrip("/"))

    meta = load(os.path.join(exp, "meta.json")) or {}
    after = by_key(load(os.path.join(exp, "sweep.json")))
    before = by_key(load(args.base))
    cfg = meta.get("config", {})

    # headline: best (hardest) difficulty solved >=90% and >=50% per size, after
    sizes = sorted({k[0] for k in after})
    headline = {}
    for s in sizes:
        pts = sorted([k[1] for k in after if k[0] == s])
        ge90 = [e for e in pts if after[(s, e)]["solve_rate"] >= 0.90]
        ge50 = [e for e in pts if after[(s, e)]["solve_rate"] >= 0.50]
        headline[s] = {"max_empty_90": max(ge90) if ge90 else None,
                       "max_empty_50": max(ge50) if ge50 else None}

    # ---- summary.md ----
    L = [f"# Experiment {expid}", "",
         f"_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_  ·  "
         f"config `{meta.get('config_path','?')}`  ·  GPU {meta.get('gpu','?')}  ·  "
         f"git `{meta.get('git_commit','?')[:8]}`", ""]
    # config highlights
    L += ["## Config", "",
          f"- model: `{cfg.get('model','?')}`, LoRA rank {cfg.get('lora_rank','?')}, "
          f"cot={cfg.get('cot', True)}",
          f"- GRPO: K={cfg.get('num_generations','?')}, "
          f"batch={cfg.get('per_device_train_batch_size','?')}, "
          f"grad_accum={cfg.get('gradient_accumulation_steps','?')}, "
          f"LR={cfg.get('learning_rate','?')}, beta={cfg.get('beta',0.04)}",
          f"- total train time: {meta.get('total_seconds',0)/60:.1f} min", ""]
    # per-stage training
    L += ["## Training (per stage)", "",
          "| stage | size | empties | steps | solved first10 → last10 | max | min |",
          "|---|---|---|---|---|---|---|"]
    for st in meta.get("stages", []):
        f10 = st.get("solved_first10"); l10 = st.get("solved_last10"); mx = st.get("solved_max")
        L.append(f"| {st['stage']} | {st['size']}x{st['size']} | {st['empties']} | "
                 f"{st['max_steps']} | {f10:.0%} → {l10:.0%} | {mx:.0%} | "
                 f"{st['seconds']/60:.1f}m |")
    # before/after
    L += ["", "## Before vs after (solve rate by difficulty)", "",
          "| puzzle | empty | before | after | Δ |", "|---|---|---|---|---|"]
    for k in sorted(after):
        a = after[k]["solve_rate"]
        b = before.get(k, {}).get("solve_rate")
        bcell = f"{b:.0%}" if b is not None else "—"
        dcell = f"{a-b:+.0%}" if b is not None else "—"
        L.append(f"| {k[0]}x{k[0]} | {k[1]} | {bcell} | {a:.0%} | {dcell} |")
    L += ["", "## Headline", ""]
    for s in sizes:
        h = headline[s]
        L.append(f"- **{s}x{s}**: solved ≥90% up to **{h['max_empty_90']}** empty cells, "
                 f"≥50% up to **{h['max_empty_50']}** empty cells.")
    summ_path = os.path.join(exp, "summary.md")
    with open(summ_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", summ_path)

    # ---- append to REGISTRY.md ----
    os.makedirs(EXPROOT, exist_ok=True)
    if not os.path.exists(REGISTRY):
        with open(REGISTRY, "w") as f:
            f.write("# Experiment registry\n\n"
                    "Each row is one training run. \"4x4≥50%\" = hardest 4x4 (most empty "
                    "cells) still solved ≥50% after training. See each `<id>/summary.md`.\n\n"
                    "| exp | date | sizes | K | LR | steps | time | 4x4 ≥90%/≥50% | 6x6 ≥50% | 8x8 ≥50% | notes |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|\n")
    sizes_str = "→".join(f"{s}" for s in sorted({st["size"] for st in meta.get("stages", [])}))
    total_steps = sum(st["max_steps"] for st in meta.get("stages", []))
    def cell(s):
        h = headline.get(s, {})
        return h.get("max_empty_50")
    row = (f"| {expid} | {datetime.datetime.now().strftime('%m-%d %H:%M')} | {sizes_str} | "
           f"{cfg.get('num_generations','?')} | {cfg.get('learning_rate','?')} | "
           f"{total_steps} | {meta.get('total_seconds',0)/60:.0f}m | "
           f"{headline.get(4,{}).get('max_empty_90','-')}/{headline.get(4,{}).get('max_empty_50','-')} | "
           f"{cell(6) if cell(6) else '-'} | {cell(8) if cell(8) else '-'} | |\n")
    with open(REGISTRY, "a") as f:
        f.write(row)
    print("appended row to", REGISTRY)


if __name__ == "__main__":
    main()
