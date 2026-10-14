"""Typed config. One YAML file = one run. Method and model files are merged on top of base.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

MethodName = Literal["lora", "qlora", "dora", "adalora", "loraplus", "full"]   # full = every weight trained


class ModelCfg(BaseModel):
    name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision: str = "main"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    max_seq_length: int = 2048
    device_map: str | None = None  # "auto" = one process spreads layers over all GPUs (naive model parallel)


class AdaLoraCfg(BaseModel):
    init_r: int = 12
    target_r: int = 8
    tinit_ratio: float = 0.10   # fraction of total steps with no pruning (warm-up)
    tfinal_ratio: float = 0.20  # fraction of total steps at the end with ranks frozen
    deltaT: int = 10            # prune every deltaT steps
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5


class QuantCfg(BaseModel):
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    upcast_non_quantized: bool = True  # False: keep fused MoE experts/embeddings in bf16 (saves memory)


class MethodCfg(BaseModel):
    name: MethodName = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    use_rslora: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    loraplus_lr_ratio: float = 16.0
    load_in_4bit: bool = False  # forced True for "qlora"; set True with dora/loraplus/adalora for Q-variants
    adalora: AdaLoraCfg = Field(default_factory=AdaLoraCfg)
    quant: QuantCfg = Field(default_factory=QuantCfg)


class DataCfg(BaseModel):
    train_file: str = "data/processed/train.jsonl"
    eval_file: str | None = "data/processed/eval.jsonl"
    packing: bool = False
    assistant_only_loss: bool = False   # needs {% generation %} markers in the chat template (most official ones lack them)
    completion_only_loss: bool = False  # works with ANY template: loss on the LAST assistant turn only (TRL prompt-completion)


class TrainCfg(BaseModel):
    output_dir: str = "outputs/run"
    epochs: float = 1.0
    max_steps: int = -1
    per_device_batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    scheduler: str = "cosine"
    warmup_steps: float = 0.03  # int = exact steps, float in [0,1) = ratio (transformers 5.x)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "none"
    seed: int = 42


class HubCfg(BaseModel):
    push: bool = False
    repo_id: str | None = None
    private: bool = True


class RunCfg(BaseModel):
    run_name: str = "peft-run"
    model: ModelCfg = Field(default_factory=ModelCfg)
    method: MethodCfg = Field(default_factory=MethodCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    hub: HubCfg = Field(default_factory=HubCfg)

    @model_validator(mode="after")
    def _checks(self) -> "RunCfg":
        m = self.method
        if m.name == "adalora" and m.adalora.target_r > m.adalora.init_r:
            raise ValueError("adalora.target_r must be <= adalora.init_r")
        if m.name == "qlora":
            m.load_in_4bit = True
        if m.name == "full" and m.load_in_4bit:
            raise ValueError("full fine-tuning cannot train 4-bit weights: use method qlora, or load_in_4bit: false")
        if self.data.assistant_only_loss and self.data.completion_only_loss:
            raise ValueError("choose ONE of data.assistant_only_loss / data.completion_only_loss")
        if m.load_in_4bit and self.model.dtype == "float32":
            raise ValueError("qlora needs a half-precision compute dtype (bfloat16/float16)")
        return self


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path, overrides: dict | None = None) -> RunCfg:
    """Merge YAML files left-to-right, then dict overrides, then validate."""
    merged: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            merged = deep_merge(merged, yaml.safe_load(f) or {})
    merged = deep_merge(merged, overrides or {})
    return RunCfg.model_validate(merged)


def parse_cli_overrides(pairs: list[str]) -> dict:
    """Turn ["train.lr=1e-4", "method.r=32"] into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {pair!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out
