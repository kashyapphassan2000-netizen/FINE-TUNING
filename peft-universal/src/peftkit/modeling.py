"""Model + tokenizer loading that works for any causal-LM repo on the Hugging Face Hub."""
from __future__ import annotations

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RunCfg
from .peft_factory import DTYPES, build_bnb_config

log = logging.getLogger("peftkit")

# Fallback template for base models that ship without one. {% generation %} markers let TRL
# compute an assistant-only loss mask. Prefer an *-Instruct checkpoint that has its own template.
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' }}"
    "{%- if message['role'] == 'assistant' %}"
    "{% generation %}{{- message['content'] + '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
    "{%- else %}{{- message['content'] + '<|im_end|>' + '\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def load_tokenizer(cfg: RunCfg):
    tok = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path, revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code
    )
    if tok.pad_token is None:
        log.warning("tokenizer has no pad token; using eos as pad")
        tok.pad_token = tok.eos_token
    if not tok.chat_template:
        log.warning("tokenizer has no chat template; installing a ChatML fallback")
        tok.chat_template = CHATML_TEMPLATE
    tok.padding_side = "right"  # training; generation code sets left padding itself
    return tok


def load_model(cfg: RunCfg):
    cuda = torch.cuda.is_available()
    dtype = DTYPES[cfg.model.dtype] if cuda else torch.float32
    kwargs = dict(
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    if cfg.method.load_in_4bit:
        if not cuda:
            raise RuntimeError("4-bit (QLoRA) loading needs an NVIDIA GPU with bitsandbytes")
        kwargs["quantization_config"] = build_bnb_config(cfg.method)
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}  # one full copy per process
    if cfg.model.device_map == "auto" and cuda:
        # single process, layers spread over every visible GPU. Do NOT combine with accelerate launch.
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    model.config.use_cache = False  # KV cache is useless during training and breaks checkpointing
    gc = cfg.train.gradient_checkpointing
    if cfg.method.load_in_4bit:
        if cfg.method.quant.upcast_non_quantized:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gc, gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model = light_kbit_prep(model, gc)
    elif gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # frozen embeddings still need to pass grads through
    return model


def light_kbit_prep(model, gradient_checkpointing: bool = True):
    """Like PEFT's prepare_model_for_kbit_training, but only 1-D tensors (norms, biases) go to fp32.
    Large unquantized tensors (fused MoE experts, embeddings, lm_head) stay in bf16.
    Trade-off: slightly less numerically robust; much less memory on MoE models."""
    for p in model.parameters():
        p.requires_grad = False
        if p.dim() == 1 and p.dtype in (torch.float16, torch.bfloat16) and p.__class__.__name__ != "Params4bit":
            p.data = p.data.float()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def list_linear_modules(model) -> dict[str, int]:
    """Unique leaf names of Linear-like layers -> count. Use to pick target_modules for any architecture."""
    counts: dict[str, int] = {}
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls in ("Linear", "Linear4bit", "Linear8bitLt") and "lm_head" not in name:
            leaf = name.split(".")[-1]
            counts[leaf] = counts.get(leaf, 0) + 1
    return counts
