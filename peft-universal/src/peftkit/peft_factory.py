"""Maps the 5 methods onto real PEFT objects. This is the only place method math is chosen."""
from __future__ import annotations

import math

import torch
from peft import AdaLoraConfig, LoraConfig, PeftConfig

from .config import MethodCfg

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def adalora_schedule(total_steps: int, a) -> tuple[int, int]:
    tinit = max(1, int(total_steps * a.tinit_ratio))
    tfinal = max(1, int(total_steps * a.tfinal_ratio))
    if tinit >= total_steps - tfinal:
        raise ValueError(
            f"AdaLoRA needs tinit < total_step - tfinal (got tinit={tinit}, tfinal={tfinal}, total={total_steps}). "
            "Train for more steps or lower tinit_ratio/tfinal_ratio."
        )
    return tinit, tfinal


def build_peft_config(m: MethodCfg, total_steps: int | None = None) -> PeftConfig:
    common = dict(
        lora_alpha=m.alpha,
        lora_dropout=m.dropout,
        target_modules=m.target_modules,
        bias=m.bias,
        task_type="CAUSAL_LM",
    )
    if m.name in ("lora", "qlora", "loraplus"):
        return LoraConfig(r=m.r, use_rslora=m.use_rslora, **common)
    if m.name == "dora":
        return LoraConfig(r=m.r, use_rslora=m.use_rslora, use_dora=True, **common)
    if m.name == "adalora":
        if not total_steps:
            raise ValueError("AdaLoRA requires the total number of optimizer steps")
        a = m.adalora
        tinit, tfinal = adalora_schedule(total_steps, a)
        return AdaLoraConfig(
            init_r=a.init_r, target_r=a.target_r, tinit=tinit, tfinal=tfinal, deltaT=a.deltaT,
            beta1=a.beta1, beta2=a.beta2, orth_reg_weight=a.orth_reg_weight, total_step=total_steps, **common,
        )
    raise ValueError(f"unknown method {m.name}")


def build_bnb_config(m: MethodCfg):
    from transformers import BitsAndBytesConfig

    q = m.quant
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.quant_type,
        bnb_4bit_use_double_quant=q.double_quant,
        bnb_4bit_compute_dtype=DTYPES[q.compute_dtype],
    )


def optimizer_cls_for(optim_name: str):
    """Optimizer class used when we must build the optimizer ourselves (LoRA+)."""
    if "8bit" in optim_name:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit if "paged" in optim_name else bnb.optim.AdamW8bit
    return torch.optim.AdamW


def total_optimizer_steps(n_examples: int, per_device_bs: int, grad_accum: int, epochs: float,
                          max_steps: int, world_size: int = 1) -> int:
    if max_steps and max_steps > 0:
        return max_steps
    batches = math.ceil(n_examples / (per_device_bs * world_size))
    per_epoch = max(1, math.ceil(batches / grad_accum))
    return max(1, math.ceil(per_epoch * epochs))
