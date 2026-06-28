#!/usr/bin/env python3
"""Curriculum GRPO training on a single GPU, with full per-experiment logging.

Each stage's puzzles are generated in-process at the configured size + clue count,
LoRA weights carry forward across stages. Everything is logged:
  <exp>/train_log_stage{i}.json   full TRL log_history (every metric, every step)
  <exp>/meta.json                 config, git commit, GPU, versions, per-stage timing
  <exp>/adapter/                  final LoRA adapter
  <exp>/adapter_stage{i}/         per-stage checkpoints

    python scripts/train.py --config configs/easy_curriculum.yaml --exp-dir experiments/exp001
"""
import argparse
import json
import os
import subprocess
import sys
import time

# IMPORTANT: patch TRL's GRPOTrainer BEFORE trl is imported anywhere, so it reuses
# Unsloth's in-process vLLM engine (colocate) instead of spawning a second engine.
from unsloth import FastLanguageModel, PatchFastRL  # noqa: E402  (must precede trl)
PatchFastRL("GRPO", FastLanguageModel)

import yaml

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sudoku_rlvr.modeling import load_model  # noqa: E402
from sudoku_rlvr.rewards import (  # noqa: E402
    format_reward, solution_reward, solved_metric,
)
from sudoku_rlvr.sudoku import DEFAULT_CLUES  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def dump_log(trainer, path):
    """Write the FULL TRL log_history (all metrics, every step) — nothing dropped."""
    rows = [rec for rec in trainer.state.log_history if "reward" in rec]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def stage_summary(rows):
    def col(k):
        return [r[k] for r in rows if k in r]
    sm = col("rewards/solved_metric/mean")
    rw = col("reward")
    def mean(x):
        return sum(x) / len(x) if x else None
    return {
        "logged_steps": len(rows),
        "solved_first10": mean(sm[:10]),
        "solved_last10": mean(sm[-10:]),
        "solved_max": max(sm) if sm else None,
        "reward_first10": mean(rw[:10]),
        "reward_last10": mean(rw[-10:]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/easy_curriculum.yaml"))
    ap.add_argument("--exp-dir", default=os.path.join(ROOT, "outputs"),
                    help="directory for this run's logs/adapters/meta")
    ap.add_argument("--resume-adapter", default=None,
                    help="path to a prior LoRA adapter to continue training from")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp = os.path.abspath(args.exp_dir)
    os.makedirs(exp, exist_ok=True)

    from datasets import Dataset
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from sudoku_rlvr.data import build_examples

    t_start = time.time()
    model, tokenizer = load_model(
        model_name=cfg["model"],
        max_seq_length=cfg["max_seq_length"],
        lora_rank=cfg["lora_rank"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        for_training=True,
    )

    resume = args.resume_adapter or cfg.get("resume_adapter")
    if resume:
        from peft import load_peft_weights, set_peft_model_state_dict
        sd = load_peft_weights(os.path.abspath(resume))
        missing = set_peft_model_state_dict(model, sd)
        print(f"resumed LoRA weights from {resume} "
              f"(loaded {len(sd)} tensors)")

    reward_funcs = [format_reward, solution_reward, solved_metric]
    reward_weights = [1.0, 1.0, 0.0]  # solved_metric is logged only, not optimized

    meta = {
        "config_path": os.path.relpath(args.config, ROOT),
        "config": cfg,
        "resume_adapter": resume,
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "stages": [],
    }

    cot = cfg.get("cot", True)
    for i, stage in enumerate(cfg["stages"]):
        n = stage["size"]
        clues = stage.get("clues")
        if clues is None:
            clues = DEFAULT_CLUES[n]
        rows = build_examples(n, stage.get("train_size", 2000),
                              seed=cfg["seed"] + i, clues=clues, cot=cot,
                              split="train")
        ds = Dataset.from_list(rows)

        targs = GRPOConfig(
            output_dir=os.path.join(exp, f"_ckpt_stage{i}"),
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            optim=cfg["optim"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_generations=cfg["num_generations"],
            max_prompt_length=cfg["max_prompt_length"],
            max_completion_length=stage["max_completion_length"],
            max_steps=stage["max_steps"],
            temperature=cfg["train_temperature"],
            beta=cfg.get("beta", 0.04),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            logging_steps=cfg["logging_steps"],
            save_steps=10_000,
            seed=cfg["seed"],
            reward_weights=reward_weights,
            report_to="none",
            use_vllm=True,
            vllm_mode="colocate",
        )
        trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                              reward_funcs=reward_funcs, args=targs, train_dataset=ds)
        empties = n * n - (clues if clues is not None else 0)
        print(f"\n===== STAGE {i}: {n}x{n} clues={clues} (~{empties} empty) "
              f"| {len(ds)} puzzles, {stage['max_steps']} steps, "
              f"K={cfg['num_generations']} =====")
        t0 = time.time()
        trainer.train()
        dt = time.time() - t0

        log = dump_log(trainer, os.path.join(exp, f"train_log_stage{i}.json"))
        model.save_lora(os.path.join(exp, f"adapter_stage{i}"))
        sm = stage_summary(log)
        sm.update({"stage": i, "size": n, "clues": clues, "empties": empties,
                   "max_steps": stage["max_steps"], "seconds": round(dt, 1)})
        meta["stages"].append(sm)
        print(f"  stage {i} done in {dt/60:.1f} min | solved {sm['solved_first10']:.0%}"
              f" -> {sm['solved_last10']:.0%} (max {sm['solved_max']:.0%})")
        # persist meta after every stage so a crash still leaves a record
        meta["total_seconds"] = round(time.time() - t_start, 1)
        with open(os.path.join(exp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    model.save_lora(os.path.join(exp, "adapter"))
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. Adapter -> {exp}/adapter")


if __name__ == "__main__":
    main()
