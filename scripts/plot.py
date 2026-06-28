#!/usr/bin/env python3
"""Charts for one experiment.

    python scripts/plot.py --exp-dir experiments/exp001 [--base results/difficulty_sweep_base.json]

Reads <exp>/meta.json, <exp>/train_log_stage*.json, <exp>/sweep.json and writes:
  <exp>/training_curves.png   reward / solved / format / completion-len / KL per stage
  <exp>/difficulty_sweep.png  before vs after solve rate by size
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GREY, GREEN, RED = "#2a7fff", "#bbbbbb", "#2ca02c", "#d62728"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def stage_logs(exp, meta):
    out = []
    paths = sorted(glob.glob(os.path.join(exp, "train_log_stage*.json")),
                   key=lambda p: int(re.search(r"stage(\d+)", p).group(1)))
    stages_meta = meta.get("stages", []) if meta else []
    for i, p in enumerate(paths):
        recs = [r for r in json.load(open(p)) if "reward" in r]
        label = "?"
        if i < len(stages_meta):
            sm = stages_meta[i]
            label = f"{sm['size']}x{sm['size']} e{sm['empties']}"
        out.append((label, recs))
    return out


def _series(stages, key):
    xs, ys, step, bounds = [], [], 0, []
    for label, recs in stages:
        for r in recs:
            if key in r:
                xs.append(step); ys.append(r[key])
            step += 1
        bounds.append((step, label))
    return xs, ys, bounds


def _bands(ax, bounds):
    prev = 0
    for i, (b, label) in enumerate(bounds):
        if i:
            ax.axvline(prev, color="grey", ls="--", lw=0.7, alpha=0.5)
        ax.axvspan(prev, b, color=BLUE if i % 2 == 0 else GREEN, alpha=0.04)
        ax.text((prev + b) / 2, 0.99, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="dimgrey")
        prev = b


def plot_training(exp, meta):
    stages = stage_logs(exp, meta)
    if not stages:
        print("no training logs; skipping training chart"); return
    panels = [("reward", "Mean reward"),
              ("rewards/solved_metric/mean", "Exact-solve rate (train)"),
              ("rewards/solution_reward/mean", "Solution reward"),
              ("rewards/format_reward/mean", "Format reward"),
              ("completion_length", "Completion length"),
              ("kl", "KL to reference")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        xs, ys, bounds = _series(stages, key)
        if xs:
            ax.plot(xs, ys, lw=1.2, color=BLUE, alpha=0.55)
            if len(ys) >= 10:
                w = max(3, len(ys) // 40)
                roll = [sum(ys[max(0, i-w):i+1]) / len(ys[max(0, i-w):i+1]) for i in range(len(ys))]
                ax.plot(xs, roll, lw=2.2, color=RED)
        _bands(ax, bounds)
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.suptitle(f"Training dynamics — {os.path.basename(exp)}", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(exp, "training_curves.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def plot_sweep(exp, base_path):
    after = load(os.path.join(exp, "sweep.json"))
    before = load(base_path)
    if not after:
        print("no sweep.json; skipping sweep chart"); return
    bk = {(r["size"], r["empties"]): r for r in before} if before else {}
    ak = {(r["size"], r["empties"]): r for r in after}
    sizes = sorted({s for s, _ in ak})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.4), squeeze=False)
    for ax, s in zip(axes[0], sizes):
        emp = sorted(e for (sz, e) in ak if sz == s)
        a = [ak[(s, e)]["solve_rate"] for e in emp]
        ax.plot(emp, a, "o-", color=BLUE, label="after")
        if bk:
            pts = [(e, bk[(s, e)]["solve_rate"]) for e in emp if (s, e) in bk]
            if pts:
                ax.plot([e for e, _ in pts], [v for _, v in pts], "o--", color=GREY, label="before")
        ax.set_title(f"{s}x{s}"); ax.set_xlabel("empty cells (harder →)")
        ax.set_ylabel("solve rate"); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Solve rate vs difficulty — {os.path.basename(exp)}", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(exp, "difficulty_sweep.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..",
                                                   "results/difficulty_sweep_base.json"))
    args = ap.parse_args()
    exp = os.path.abspath(args.exp_dir)
    meta = load(os.path.join(exp, "meta.json"))
    plot_training(exp, meta)
    plot_sweep(exp, args.base)


if __name__ == "__main__":
    main()
