"""Model loading + generation helpers (Unsloth + vLLM).

Isolated here so training and evaluation share identical model setup. Imports of
unsloth/vllm are deferred to call time so the pure-Python core stays importable
without the GPU stack.
"""
from __future__ import annotations

from typing import List, Optional

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_model(model_name: str, max_seq_length: int, lora_rank: int,
               gpu_memory_utilization: float = 0.6, load_in_4bit: bool = True,
               for_training: bool = True):
    """Load a 4-bit base model with vLLM fast-inference; attach LoRA if training.

    NOTE: for GRPO training, call `unsloth.PatchFastRL("GRPO", FastLanguageModel)`
    at the very top of the entrypoint *before* importing TRL, so TRL's GRPOTrainer
    is rewritten to reuse the in-process vLLM engine (colocate) instead of spawning
    a second engine / expecting a vLLM server. See scripts/train.py.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,                # vLLM generation backend
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=LORA_TARGETS,
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    return model, tokenizer


def generate(model, tokenizer, chats: List[list], max_new_tokens: int,
             temperature: float = 0.0, lora_path: Optional[str] = None) -> List[str]:
    """Batch-generate completions for a list of conversational prompts."""
    from vllm import SamplingParams

    prompts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in chats
    ]
    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0 else 0.95,
        max_tokens=max_new_tokens,
    )
    lora_request = model.load_lora(lora_path) if lora_path else None
    outputs = model.fast_generate(prompts, sampling_params=sampling,
                                  lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]
