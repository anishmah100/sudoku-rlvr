#!/usr/bin/env python3
"""Quick training-health check from a run log — for close monitoring.

    python scripts/monitor.py <logfile>

Prints current stage/step, and the solved-rate & reward trend so problems
(flat/declining solved, format collapse, KL blowup) are caught early.
"""
import re
import sys


def main():
    path = sys.argv[1]
    txt = open(path, encoding="utf-8", errors="ignore").read().replace("\r", "\n")
    lines = txt.splitlines()

    stages = [l.strip() for l in lines if "STAGE" in l]
    print("stages seen:")
    for s in stages[-6:]:
        print("  " + s.replace("=====", "").strip())

    prog = re.findall(r"\b(\d+)/(\d+) \[[^\]]+\]", txt)
    if prog:
        print(f"progress: {prog[-1][0]}/{prog[-1][1]}")

    def col(key):
        return [float(m.group(1)) for l in lines
                for m in [re.search(key, l)] if m]
    sm = col(r"'rewards/solved_metric/mean': ([0-9.]+)")
    rw = col(r"'reward': ([-0-9.]+)")
    fm = col(r"'rewards/format_reward/mean': ([0-9.]+)")
    kl = col(r"'kl': ([0-9.eE+-]+)")
    n = len(sm)
    print(f"logged steps: {n}")
    if n:
        def tail(x, k=20):
            x = x[-k:]
            return sum(x) / len(x) if x else 0.0
        print(f"  solved  last20={tail(sm):.3f}  max={max(sm):.3f}")
        print(f"  reward  last20={tail(rw):+.3f}")
        if fm:
            print(f"  format  last20={tail(fm):.3f} (max 0.20)")
        if kl:
            print(f"  kl      last20={tail(kl):.4f}")
    errs = [l for l in lines if re.search(r"Traceback|Error:|OutOfMemory|CUDA error", l)]
    if errs:
        print("!! ERRORS:")
        for e in errs[-5:]:
            print("   " + e[:120])


if __name__ == "__main__":
    main()
