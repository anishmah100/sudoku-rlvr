#!/usr/bin/env python3
"""Build the consolidated frontier charts and table across all experiments.

Outputs:
  assets/frontier.png             solve rate vs empty cells, one line per board size
  assets/frontier_8x8.png         8x8 frontier advancing across exp4 -> exp7
  results/frontier.md             the combined table

Run after the per-experiment sweeps exist.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXP = os.path.join(ROOT, "experiments")
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else []


def size_curve(size, *paths):
    """Merge sweep files, keep rows for `size`, return {empties: solve_rate} sorted."""
    pts = {}
    for p in paths:
        for r in load(p):
            if r["size"] == size:
                pts[r["empties"]] = r["solve_rate"]
    return dict(sorted(pts.items()))


# Final per-size curves (best adapter per size; 8x8 merges the e1-3 probe).
FINAL = {
    4: size_curve(4, os.path.join(EXP, "exp1_4x4/sweep.json")),
    6: size_curve(6, os.path.join(EXP, "exp8_6x6_hard/sweep.json")),
    8: size_curve(8, os.path.join(EXP, "exp7_8x8_ceiling/sweep.json"),
                  os.path.join(EXP, "exp7_easy8.json")),
}

COLORS = {4: "#2a7fff", 6: "#2ca02c", 8: "#d62728"}


def plot_frontier():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for size, curve in FINAL.items():
        if not curve:
            continue
        xs, ys = list(curve), list(curve.values())
        ax.plot(xs, ys, "o-", color=COLORS[size], lw=2, label=f"{size}x{size}")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("empty cells (harder →)")
    ax.set_ylabel("exact-solve rate (held-out, greedy)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Trained solve rate vs difficulty, by board size\n"
                 "(base model ≈ 0% at every point)", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(title="board")
    fig.tight_layout()
    out = os.path.join(ASSETS, "frontier.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_8x8_progression():
    runs = [
        ("exp4 (transcription warm-up)", "#bdd7e7",
         "exp4_8x8_transcribe/sweep.json", "exp4_easy8.json"),
        ("exp5 (mid)", "#6baed6", "exp5_8x8_mid/sweep.json", "exp5_easy8.json"),
        ("exp6 (hard-mid)", "#3182bd", "exp6_8x8_hard/sweep.json", "exp6_easy8.json"),
        ("exp7 (ceiling)", "#08519c", "exp7_8x8_ceiling/sweep.json", "exp7_easy8.json"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, color, sweep, easy in runs:
        curve = size_curve(8, os.path.join(EXP, sweep), os.path.join(EXP, easy))
        if not curve:
            continue
        ax.plot(list(curve), list(curve.values()), "o-", color=color, lw=2, label=label)
    ax.set_xlabel("empty cells (harder →)")
    ax.set_ylabel("8x8 exact-solve rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("8x8 frontier advancing with successive training runs",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ASSETS, "frontier_8x8.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def write_table():
    base = {(r["size"], r["empties"]): r["solve_rate"]
            for r in load(os.path.join(RESULTS, "difficulty_sweep_base.json"))}
    empties = sorted({e for c in FINAL.values() for e in c})
    lines = ["# Frontier (held-out exact-solve rate, greedy)", "",
             "Base model is ~0% at every point. Trained columns use the best adapter per",
             "size (4x4 exp1, 6x6 exp3+exp8, 8x8 exp4-exp7).", "",
             "| empty cells | 4x4 | 6x6 | 8x8 |", "|---|---|---|---|"]
    for e in empties:
        row = [f"{e}"]
        for s in (4, 6, 8):
            v = FINAL[s].get(e)
            row.append(f"{v:.0%}" if v is not None else "–")
        lines.append("| " + " | ".join(row) + " |")
    out = os.path.join(RESULTS, "frontier.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    plot_frontier()
    plot_8x8_progression()
    write_table()


if __name__ == "__main__":
    main()
