"""STEP 14: MODEL COMPATIBILITY REPORT for any model.

python -m uadapt.report --model PATH_OR_HF_ID [--head causal-lm|seq-cls|seq2seq|ctc|image-cls|base]
python -m uadapt.report --custom my_model.py:MyModel --input "x=float:2,3,32,32"

HF ids need only config.json (built on meta/CPU with random weights: architecture, not quality).
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import torch

from .adapter import UniversalModelAdapter


def sample_for(model, config) -> dict:
    """A tiny valid input built from the forward signature and config. Custom models: pass --input."""
    sig = UniversalModelAdapter(model, {}).identify_inputs()
    d = config.to_dict() if hasattr(config, "to_dict") else {}
    vocab = d.get("vocab_size") or 100
    b = {}
    if "input_ids" in sig:
        b["input_ids"] = torch.randint(0, min(vocab, 100), (1, 8))
    if "pixel_values" in sig:
        vc = d.get("vision_config") or d
        s, c = vc.get("image_size", 32), vc.get("num_channels", 3)
        b["pixel_values"] = torch.randn(1, c, s if isinstance(s, int) else s[0], s if isinstance(s, int) else s[1])
    if "input_values" in sig:
        b["input_values"] = torch.randn(1, 4000)
    if "input_features" in sig:
        b["input_features"] = torch.randn(1, d.get("num_mel_bins", 80), 2 * d.get("max_source_positions", 1500))
    if "decoder_input_ids" in sig and "input_ids" in b or "decoder_input_ids" in sig and "input_features" in b:
        b["decoder_input_ids"] = torch.zeros(1, 4, dtype=torch.long)
    return b


def build(model_id: str, head: str):
    from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForCTC,
                              AutoModelForImageClassification, AutoModelForSeq2SeqLM,
                              AutoModelForSequenceClassification)
    cfg = AutoConfig.from_pretrained(model_id)
    cls = {"causal-lm": AutoModelForCausalLM, "seq-cls": AutoModelForSequenceClassification,
           "seq2seq": AutoModelForSeq2SeqLM, "ctc": AutoModelForCTC, "image-cls": AutoModelForImageClassification,
           "base": AutoModel}[head]
    return cls.from_config(cfg), cfg


def report(model, config, batch, vram_gb: float = 24.0) -> dict:
    a = UniversalModelAdapter(model, batch, config)
    info, cls = a.inspect(), a.classify()
    t = a.discover_trainable_modules()
    loss = a.identify_loss()
    strat = a.recommend_peft_strategy(vram_gb, task="?")
    d = config.to_dict() if hasattr(config, "to_dict") else {}
    risks = []
    if "moe" in cls["labels"]:
        risks.append("MoE: fused experts are not bitsandbytes-quantized; LoRA on all experts is huge (chapter 22)")
    if d.get("quantization_config"):
        risks.append("pre-quantized checkpoint: make a bf16 copy for training")
    if not t["lora_targets"]:
        risks.append("no Linear LoRA targets found: selective unfreezing or custom adapters")
    if not loss["model_returns_loss"]:
        risks.append("forward() returns no loss for this input: a custom loss_fn is required")
    custom = []
    if "custom" in cls["labels"]:
        custom.append("training loop (uadapt.loop) and a loss_fn")
    if any(k in cls["labels"] for k in ("vision", "audio", "multimodal")):
        custom.append("processor + collator for the modality (STEP 15)")
    if "decoder-only" not in cls["labels"]:
        custom.append("the peftkit/TRL SFT path does not apply as-is: use uadapt.loop or the family's Trainer")
    return {
        "Architecture": type(model).__name__, "Classification": cls["labels"], "Evidence": cls["evidence"],
        "Parameter Count": info["params"]["total"], "Trainable Parameters": info["params"]["trainable"],
        "Input Types": info["signature"]["parameters"], "Output Types": a.identify_outputs(),
        "Forward Signature": list(info["signature"]["parameters"]),
        "Loss": loss, "Module Inventory": info["inventory"],
        "Potential LoRA Targets (leaf names)": t["leaf_names"], "LoRA target count": len(t["lora_targets"]),
        "Do not adapt": t["do_not_adapt"][:12], "Train fully or freeze (heads)": t["train_fully_or_freeze"],
        "Conv candidates": t["conv_candidates"][:8],
        "Recommended PEFT (ranked)": [o["strategy"] for o in strat["ranked"]][:4],
        "Memory Estimate (GB)": strat["memory_gb"], "Risks": risks, "Required Custom Code": custom,
        "Recommended serving": a.serve()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Model compatibility report for any architecture")
    ap.add_argument("--model", default=None)
    ap.add_argument("--head", default="base", choices=["causal-lm", "seq-cls", "seq2seq", "ctc", "image-cls", "base"])
    ap.add_argument("--custom", default=None, help="file.py:ClassName (constructed with no arguments)")
    ap.add_argument("--input", default=None, help='for --custom: "name=float:1,3,32,32 other=long:1,8"')
    ap.add_argument("--vram-gb", type=float, default=24.0)
    a = ap.parse_args(argv)
    if a.custom:
        path, cls = a.custom.split(":")
        spec = importlib.util.spec_from_file_location("custom_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model, cfg = getattr(mod, cls)(), None
        batch = {}
        for part in (a.input or "").split():
            k, v = part.split("=")
            kind, shape = v.split(":")
            dims = tuple(int(x) for x in shape.split(","))
            batch[k] = torch.randn(dims) if kind == "float" else torch.randint(0, 10, dims)
    else:
        model, cfg = build(a.model, a.head)
        batch = sample_for(model, cfg)
    print("MODEL COMPATIBILITY REPORT")
    for k, v in report(model, cfg, batch, a.vram_gb).items():
        print(f"{k}: {json.dumps(v, default=str) if not isinstance(v, str) else v}")


if __name__ == "__main__":
    main()
