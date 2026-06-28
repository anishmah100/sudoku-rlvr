#!/usr/bin/env python3
"""Curriculum GRPO training on a single GPU.

Runs stages 4x4 -> 6x6 -> 9x9, carrying LoRA weights forward. Per-step metrics
(reward, format reward, solution reward, solved rate) are dumped to
results/train_log_stage{i}_{n}x{n}.json for plotting. Final adapter is saved to
outputs/adapter.

    source .venv/bin/activate && source env.sh
    python scripts/train.py --config configs/curriculum.yaml
"""
import argparse
import json
import os
import sys

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

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
OUTPUTS = os.path.join(ROOT, "outputs")


def dump_log(trainer, path):
    keep = ("loss", "reward", "reward_std", "kl",
            "rewards/format_reward/mean", "rewards/solution_reward/mean",
            "rewards/solved_metric/mean", "completion_length", "step")
    rows = []
    for rec in trainer.state.log_history:
        row = {k: rec[k] for k in keep if k in rec}
        if "step" not in row and "epoch" in rec:
            row["step"] = rec.get("step")
        if row:
            rows.append(rec)  # keep full record; plot picks what it needs
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/curriculum.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(OUTPUTS, exist_ok=True)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from sudoku_rlvr.data import build_examples

    model, tokenizer = load_model(
        model_name=cfg["model"],
        max_seq_length=cfg["max_seq_length"],
        lora_rank=cfg["lora_rank"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        for_training=True,
    )

    reward_funcs = [format_reward, solution_reward, solved_metric]
    reward_weights = [1.0, 1.0, 0.0]  # solved_metric is logged only, not optimized

    cot = cfg.get("cot", True)
    for i, stage in enumerate(cfg["stages"]):
        n = stage["size"]
        clues = stage.get("clues")  # None -> DEFAULT_CLUES[n]
        # Generate this stage's puzzles in-process so per-stage difficulty (clue
        # count) is fully controlled by the curriculum config.
        rows = build_examples(n, stage.get("train_size", 2000),
                              seed=cfg["seed"] + i, clues=clues, cot=cot)
        ds = Dataset.from_list(rows)
        tag = f"stage{i}_{n}x{n}_c{clues if clues is not None else 'def'}"
        stage_out = os.path.join(OUTPUTS, tag)

        targs = GRPOConfig(
            output_dir=stage_out,
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
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            seed=cfg["seed"],
            reward_weights=reward_weights,
            report_to="none",
            use_vllm=True,
            vllm_mode="colocate",  # use Unsloth's in-process engine, not a server
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=targs,
            train_dataset=ds,
        )
        print(f"\n===== STAGE {i}: {n}x{n} clues={clues} "
              f"({len(ds)} puzzles, {stage['max_steps']} steps) =====")
        trainer.train()
        dump_log(trainer, os.path.join(RESULTS, f"train_log_stage{i}_{n}x{n}.json"))
        model.save_lora(os.path.join(OUTPUTS, f"adapter_stage{i}_{n}x{n}"))

    model.save_lora(os.path.join(OUTPUTS, "adapter"))
    print("\nDone. Final adapter -> outputs/adapter")


if __name__ == "__main__":
    main()
