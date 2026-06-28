#!/usr/bin/env python3
"""Build the full chart + table + observations suite for the study.

Inputs (whatever exists):
  results/train_log_stage*.json   -> training-dynamics charts
  results/eval_base.json          -> baseline metrics
  results/eval_trained.json       -> post-training metrics

Outputs (assets/*.png, results/*):
  training_curves.png      6-panel training dynamics across the curriculum
  reward_components.png    reward decomposition over training
  before_after.png         grouped before/after bars (4 metrics x sizes)
  improvement_delta.png    per-size improvement (after - before)
  dashboard.png            one-page summary
  results/summary.csv      full metric table
  results/comparison.md    before/after table with deltas
  results/observations.md  auto-generated narrative observations
"""
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "assets")

BLUE, GREY, GREEN, RED = "#2a7fff", "#bbbbbb", "#2ca02c", "#d62728"

EVAL_METRICS = [
    ("solve_rate", "Exact solve rate"),
    ("frac_correct", "Cell accuracy"),
    ("format_rate", "Valid format rate"),
    ("givens_ok_rate", "Givens preserved"),
]


# --------------------------------------------------------------------------- #
# Training logs
# --------------------------------------------------------------------------- #
def stage_logs():
    """Return [(stage_index, size, [records]), ...] sorted by stage index."""
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "train_log_stage*.json"))):
        m = re.search(r"stage(\d+)_(\d+)x\d+\.json", os.path.basename(p))
        if not m:
            continue
        idx, size = int(m.group(1)), int(m.group(2))
        with open(p) as f:
            recs = [r for r in json.load(f) if "reward" in r]
        out.append((idx, size, recs))
    return sorted(out, key=lambda t: t[0])


def _series(stages, key):
    xs, ys, step, bounds = [], [], 0, []
    for _, size, recs in stages:
        for r in recs:
            if key in r:
                xs.append(step)
                ys.append(r[key])
            step += 1
        bounds.append((step, size))
    return xs, ys, bounds


def _bands(ax, bounds):
    prev = 0
    for i, (b, size) in enumerate(bounds):
        if i:
            ax.axvline(prev, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax.axvspan(prev, b, color=BLUE if i % 2 == 0 else GREEN, alpha=0.04)
        ax.text((prev + b) / 2, 0.98, f"{size}x{size}", transform=
                ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="dimgrey")
        prev = b


def plot_training():
    stages = stage_logs()
    if not stages:
        print("no training logs yet; skipping training charts")
        return
    panels = [
        ("reward", "Mean reward"),
        ("rewards/solved_metric/mean", "Exact-solve rate (train)"),
        ("rewards/solution_reward/mean", "Solution reward"),
        ("rewards/format_reward/mean", "Format reward"),
        ("completion_length", "Completion length (tokens)"),
        ("kl", "KL to reference"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        xs, ys, bounds = _series(stages, key)
        if xs:
            ax.plot(xs, ys, lw=1.5, color=BLUE)
            # rolling mean
            if len(ys) >= 10:
                w = max(3, len(ys) // 30)
                roll = [sum(ys[max(0, i - w):i + 1]) / len(ys[max(0, i - w):i + 1])
                        for i in range(len(ys))]
                ax.plot(xs, roll, lw=2.2, color=RED, alpha=0.8, label=f"rolling({w})")
                ax.legend(fontsize=7, loc="best")
        _bands(ax, bounds)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    fig.suptitle("GRPO curriculum training dynamics — Qwen2.5-3B on one RTX 4090",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(ASSETS, "training_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_reward_components():
    stages = stage_logs()
    if not stages:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    for key, color, label in [
        ("rewards/format_reward/mean", "#ff7f0e", "format"),
        ("rewards/solution_reward/mean", BLUE, "solution"),
        ("reward", "black", "total"),
    ]:
        xs, ys, bounds = _series(stages, key)
        if xs:
            ax.plot(xs, ys, lw=1.6, color=color, label=label, alpha=0.85)
    _bands(ax, bounds)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_title("Reward decomposition over training", fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ASSETS, "reward_components.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Eval (before/after)
# --------------------------------------------------------------------------- #
def load_eval(tag):
    p = os.path.join(RESULTS, f"eval_{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def plot_before_after(base, trained):
    if not base or not trained:
        print("need eval_base.json + eval_trained.json for before/after; skipping")
        return
    sizes = sorted(base["by_size"], key=int)
    fig, axes = plt.subplots(1, len(EVAL_METRICS), figsize=(5 * len(EVAL_METRICS), 4.4))
    x = range(len(sizes))
    w = 0.38
    for ax, (key, title) in zip(axes, EVAL_METRICS):
        b = [base["by_size"][s][key] for s in sizes]
        t = [trained["by_size"][s][key] for s in sizes]
        ax.bar([i - w / 2 for i in x], b, w, label="before", color=GREY)
        ax.bar([i + w / 2 for i in x], t, w, label="after", color=BLUE)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for i, (bb, tt) in enumerate(zip(b, t)):
            ax.text(i - w / 2, bb + 0.02, f"{bb:.0%}", ha="center", fontsize=8)
            ax.text(i + w / 2, tt + 0.02, f"{tt:.0%}", ha="center", fontsize=8,
                    fontweight="bold")
    axes[0].legend(loc="upper right")
    fig.suptitle("Before vs after GRPO on held-out Sudoku (greedy decoding)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(ASSETS, "before_after.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_improvement(base, trained):
    if not base or not trained:
        return
    sizes = sorted(base["by_size"], key=int)
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.2
    for j, (key, title) in enumerate(EVAL_METRICS):
        deltas = [trained["by_size"][s][key] - base["by_size"][s][key] for s in sizes]
        ax.bar([i + (j - 1.5) * w for i in range(len(sizes))], deltas, w, label=title)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_ylabel("Δ (after − before)")
    ax.set_title("Improvement from GRPO, by board size and metric", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ASSETS, "improvement_delta.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_dashboard(base, trained):
    stages = stage_logs()
    if not (base and trained and stages):
        return
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :2])
    xs, ys, bounds = _series(stages, "reward")
    ax.plot(xs, ys, color=BLUE, lw=1.4)
    _bands(ax, bounds)
    ax.set_title("Mean reward over training", fontweight="bold")
    ax.set_xlabel("step"); ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    sizes = sorted(base["by_size"], key=int)
    b = [base["by_size"][s]["solve_rate"] for s in sizes]
    t = [trained["by_size"][s]["solve_rate"] for s in sizes]
    x = range(len(sizes)); w = 0.38
    ax2.bar([i - w / 2 for i in x], b, w, color=GREY, label="before")
    ax2.bar([i + w / 2 for i in x], t, w, color=BLUE, label="after")
    ax2.set_xticks(list(x)); ax2.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax2.set_title("Exact solve rate", fontweight="bold"); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05); ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :])
    xs, ys, bounds = _series(stages, "rewards/solved_metric/mean")
    ax3.plot(xs, ys, color=GREEN, lw=1.4)
    _bands(ax3, bounds)
    ax3.set_title("Exact-solve rate during training", fontweight="bold")
    ax3.set_xlabel("step"); ax3.grid(alpha=0.3); ax3.set_ylim(0, 1.02)

    fig.suptitle("Sudoku-RLVR — study dashboard (Qwen2.5-3B, single RTX 4090)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(ASSETS, "dashboard.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Tables + observations
# --------------------------------------------------------------------------- #
def write_tables(base, trained):
    if not (base and trained):
        return
    sizes = sorted(base["by_size"], key=int) + ["overall"]
    keys = ["solve_rate", "frac_correct", "format_rate", "givens_ok_rate", "mean_reward"]
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["size", "phase"] + keys)
        for s in sizes:
            for tag, rep in (("before", base), ("after", trained)):
                src = rep["overall"] if s == "overall" else rep["by_size"][s]
                wr.writerow([s, tag] + [f"{src[k]:.4f}" for k in keys])
    print("wrote results/summary.csv")

    md = ["# Before vs After (GRPO curriculum)", "",
          "| Size | Metric | Before | After | Δ |", "|---|---|---|---|---|"]
    for s in sizes:
        b = base["overall"] if s == "overall" else base["by_size"][s]
        t = trained["overall"] if s == "overall" else trained["by_size"][s]
        for k in ["solve_rate", "frac_correct", "format_rate", "givens_ok_rate"]:
            label = "overall" if s == "overall" else f"{s}x{s}"
            md.append(f"| {label} | {k} | {b[k]:.1%} | {t[k]:.1%} | {(t[k]-b[k]):+.1%} |")
    with open(os.path.join(RESULTS, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote results/comparison.md")


def write_observations(base, trained):
    if not (base and trained):
        return
    sizes = sorted(base["by_size"], key=int)
    L = ["# Observations", "",
         "Auto-generated from `results/eval_base.json` and `results/eval_trained.json`.",
         ""]

    ob, ot = base["overall"], trained["overall"]
    L.append(f"- **Overall exact-solve rate** moved from **{ob['solve_rate']:.1%}** "
             f"to **{ot['solve_rate']:.1%}** "
             f"({ot['solve_rate']-ob['solve_rate']:+.1%}).")
    L.append(f"- **Overall cell accuracy** moved from **{ob['frac_correct']:.1%}** "
             f"to **{ot['frac_correct']:.1%}** "
             f"({ot['frac_correct']-ob['frac_correct']:+.1%}).")

    # biggest solve-rate gain
    gains = {s: trained["by_size"][s]["solve_rate"] - base["by_size"][s]["solve_rate"]
             for s in sizes}
    best = max(gains, key=gains.get)
    L.append(f"- **Largest exact-solve gain** at **{best}x{best}**: "
             f"{base['by_size'][best]['solve_rate']:.1%} → "
             f"{trained['by_size'][best]['solve_rate']:.1%} ({gains[best]:+.1%}).")

    # format adherence
    fb = sum(base["by_size"][s]["format_rate"] for s in sizes) / len(sizes)
    ft = sum(trained["by_size"][s]["format_rate"] for s in sizes) / len(sizes)
    L.append(f"- **Format adherence** (parseable grid) averaged "
             f"{fb:.1%} → {ft:.1%} across sizes.")

    # difficulty falloff
    L.append("- **Difficulty scaling:** solve rate after training by size — " +
             ", ".join(f"{s}x{s}: {trained['by_size'][s]['solve_rate']:.0%}"
                       for s in sizes) +
             ". As expected, larger boards remain harder on a single 4090.")

    # givens
    gt = sum(trained["by_size"][s]["givens_ok_rate"] for s in sizes) / len(sizes)
    L.append(f"- **Constraint respect:** trained model preserves the given clues "
             f"in {gt:.1%} of attempts on average (reward penalizes overwriting).")

    stages = stage_logs()
    if stages:
        first = stages[0][2][0].get("reward") if stages[0][2] else None
        last = stages[-1][2][-1].get("reward") if stages[-1][2] else None
        if first is not None and last is not None:
            L.append(f"- **Training reward** rose from {first:.3f} (first logged step) "
                     f"to {last:.3f} (last), across "
                     f"{sum(len(r) for _,_,r in stages)} logged steps.")

    L += ["",
          "> Caveat: single-seed, single-GPU case study. Default 9x9 puzzles "
          "(40 clues) are largely constraint-propagation solvable; harder bands "
          "are a separate, harder regime."]
    with open(os.path.join(RESULTS, "observations.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/observations.md")


def plot_difficulty_sweep():
    """Solve rate vs difficulty (empty cells), base vs trained, per board size."""
    bp = os.path.join(RESULTS, "difficulty_sweep_base.json")
    tp = os.path.join(RESULTS, "difficulty_sweep_trained.json")
    if not (os.path.exists(bp) and os.path.exists(tp)):
        print("need difficulty_sweep_{base,trained}.json; skipping sweep chart")
        return
    base = {(r["size"], r["empties"]): r for r in json.load(open(bp))}
    trained = {(r["size"], r["empties"]): r for r in json.load(open(tp))}
    sizes = sorted({s for s, _ in base})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.4), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        emp = sorted(e for s, e in base if s == size)
        b = [base[(size, e)]["solve_rate"] for e in emp]
        t = [trained[(size, e)]["solve_rate"] for e in emp]
        ax.plot(emp, b, "o--", color=GREY, label="before (base)")
        ax.plot(emp, t, "o-", color=BLUE, label="after (curriculum)")
        ax.set_title(f"{size}x{size}")
        ax.set_xlabel("empty cells (harder →)")
        ax.set_ylabel("exact solve rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Solve rate vs difficulty — before vs after the 4×4 curriculum",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, "difficulty_sweep.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def write_sweep_tables():
    bp = os.path.join(RESULTS, "difficulty_sweep_base.json")
    tp = os.path.join(RESULTS, "difficulty_sweep_trained.json")
    if not (os.path.exists(bp) and os.path.exists(tp)):
        return
    base = {(r["size"], r["empties"]): r for r in json.load(open(bp))}
    trained = {(r["size"], r["empties"]): r for r in json.load(open(tp))}
    keys = sorted(base)

    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["size", "empties", "clues", "phase", "solve_rate", "frac_correct", "format_rate"])
        for k in keys:
            for tag, src in (("before", base), ("after", trained)):
                if k in src:
                    r = src[k]
                    wr.writerow([k[0], k[1], k[0]*k[0]-k[1], tag,
                                 f"{r['solve_rate']:.4f}", f"{r['frac_correct']:.4f}",
                                 f"{r['format_rate']:.4f}"])
    print("wrote results/summary.csv")

    md = ["# Before vs After — solve rate by difficulty", "",
          "Exact-solve rate on 100 fresh puzzles per cell, greedy decoding. The 4×4",
          "curriculum was the only thing trained; 6×6/9×9 rows show zero-shot transfer.",
          "", "| Puzzle | Empty cells | Before | After | Δ |", "|---|---|---|---|---|"]
    for k in keys:
        if k in trained:
            b, t = base[k]["solve_rate"], trained[k]["solve_rate"]
            md.append(f"| {k[0]}x{k[0]} | {k[1]} | {b:.0%} | {t:.0%} | {t-b:+.0%} |")
    with open(os.path.join(RESULTS, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote results/comparison.md")

    # observations
    four = [(k[1], base[k]["solve_rate"], trained[k]["solve_rate"]) for k in keys if k[0] == 4]
    L = ["# Observations", "",
         "Auto-generated from the difficulty sweep (`results/difficulty_sweep_*.json`).", ""]
    if four:
        L.append(f"- **4×4 mastery:** exact-solve rate on the trivial case (1 empty) rose "
                 f"{four[0][1]:.0%} → {four[0][2]:.0%}; on full 4×4 ({four[-1][0]} empties) "
                 f"{four[-1][1]:.0%} → {four[-1][2]:.0%}.")
        L.append("- **Graceful difficulty falloff:** after training, solve rate decays "
                 "smoothly with empties — " +
                 ", ".join(f"{e}→{t:.0%}" for e, _, t in four) + ".")
    trans = [(k, base[k]["solve_rate"], trained[k]["solve_rate"]) for k in keys
             if k[0] != 4 and trained[k]["solve_rate"] - base[k]["solve_rate"] > 0.02]
    if trans:
        L.append("- **Zero-shot transfer to larger boards (never trained):** " +
                 ", ".join(f"{k[0]}x{k[0]}/{k[1]}-empty {b:.0%}→{t:.0%}" for k, b, t in trans) + ".")
    L.append("- **Baseline reminder:** the untrained model solves ~0% of all but the "
             "most trivial instances, so essentially the entire after-curve is new capability.")
    L += ["", "> Scope: single RTX 4090, single seed, Qwen2.5-3B + LoRA, CoT decoding. "
          "The curriculum trains 4×4 only; larger boards are out-of-distribution."]
    with open(os.path.join(RESULTS, "observations.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results/observations.md")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    plot_difficulty_sweep()
    write_sweep_tables()
    plot_training()
    plot_reward_components()
    base, trained = load_eval("base"), load_eval("trained")
    plot_before_after(base, trained)
    plot_improvement(base, trained)
    plot_dashboard(base, trained)
    write_tables(base, trained)
    write_observations(base, trained)


if __name__ == "__main__":
    main()
