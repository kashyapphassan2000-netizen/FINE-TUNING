"""Model compatibility layer: can THIS model be fine-tuned with THIS template, and how?

Answers the honest version of "can I fine-tune any model?":
  decoder-only causal LM  -> yes, this template
  encoder-only / encoder-decoder / vision-language / multimodal -> different pipeline (told, not hidden)
  MoE -> yes, but choose targets deliberately
  custom architecture (remote code) -> manual integration, with the exact steps printed
"""
from __future__ import annotations

import argparse
import json

from .archs import exact_counts, load_config_dict, spec_from_config, text_config
from .data.templates import has_generation_markers

VERDICT_OK = "SUPPORTED"
VERDICT_ADV = "SUPPORTED-ADVANCED"
VERDICT_OTHER = "OTHER-PIPELINE"
VERDICT_MANUAL = "MANUAL-INTEGRATION"


def model_kind(cfg: dict) -> str:
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)

    mt = cfg.get("model_type", "")
    archs = " ".join(cfg.get("architectures") or [])
    if cfg.get("is_encoder_decoder") or mt in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        return "encoder-decoder"
    if any(k in cfg for k in ("vision_config", "image_token_index", "audio_config")) or "Vision" in archs:
        return "multimodal"
    if mt in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES or "ForCausalLM" in archs:
        return "decoder-only"
    if "ForMaskedLM" in archs or "ForSequenceClassification" in archs:
        return "encoder-only"
    return "unknown"


def check(model: str, trust_remote_code: bool = False, check_tokenizer: bool = True) -> dict:
    cfg = load_config_dict(model, trust_remote_code)
    kind = model_kind(cfg)
    spec = spec_from_config(cfg)
    counts = exact_counts(cfg, trust_remote_code, lora_r=16)
    report = {"model": model, "kind": kind, "model_type": spec.model_type, "architectures": spec.architectures,
              "family": spec.family, "attention": spec.attention, "blockers": [], "warnings": [], "actions": []}
    if kind == "decoder-only":
        report["verdict"] = VERDICT_ADV if spec.family == "moe" else VERDICT_OK
    elif kind in ("encoder-decoder", "encoder-only", "multimodal"):
        report["verdict"] = VERDICT_OTHER
        report["blockers"].append(f"{kind} model: this template trains causal decoder-only LMs with SFTTrainer. "
                                  "Use the matching trainer/collator for that family instead.")
    else:
        report["verdict"] = VERDICT_MANUAL
        report["blockers"].append("architecture not recognised by the installed transformers")

    if counts.get("error") or not counts.get("total"):
        report["warnings"].append("model could not be built locally (custom code?): "
                                  f"{counts.get('error', 'no counts')}. Sizing falls back to formulas; "
                                  "run with --trust-remote-code only after reading the repo's .py files")
        if report["verdict"] == VERDICT_OK:
            report["verdict"] = VERDICT_MANUAL
    else:
        report["params_total"] = counts["total"]
        report["params_active"] = counts["active"]
        report["target_modules_all"] = counts["linear_names"]
        report["target_modules_attention"] = counts["attention_linear_names"]
        report["fused_expert_tensors"] = counts["fused_expert_params"]
        report["lora_trainable_all_linear"] = counts.get("lora_trainable_all_linear")
        report["lora_trainable_attention"] = counts.get("lora_trainable_attention")
    if spec.family == "moe":
        report["warnings"].append("MoE: `all-linear` adapts every expert (can be billions of trainable params). "
                                  "Prefer attention-only targets; bitsandbytes does not quantize fused experts.")
    if spec.quantization:
        report["warnings"].append(f"pre-quantized checkpoint ({spec.quantization}): make a bf16 copy for PEFT training")
    if trust_remote_code:
        report["warnings"].append("trust_remote_code=True executes code from the repo. Read it first.")

    if check_tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
            report["tokenizer"] = {
                "vocab_size": len(tok), "has_chat_template": bool(tok.chat_template),
                "eos": tok.eos_token, "pad": tok.pad_token, "bos": tok.bos_token,
                "generation_markers": has_generation_markers(tok.chat_template),
            }
            if not tok.chat_template:
                report["warnings"].append("no chat template: base model. Prefer an -Instruct checkpoint, or the "
                                          "template installs a ChatML fallback which the model must learn")
            if tok.pad_token is None:
                report["warnings"].append("no pad token: the template will use eos as pad")
            if tok.chat_template and "generation" not in tok.chat_template:
                report["warnings"].append("chat template has no {% generation %} markers: assistant_only_loss "
                                          "will fail; use peftkit templates or completion-only loss")
        except Exception as exc:
            report["warnings"].append(f"tokenizer not checked: {type(exc).__name__}: {str(exc)[:120]}")

    if report["verdict"] in (VERDICT_OK, VERDICT_ADV):
        targets = (report.get("target_modules_attention") if spec.family == "moe"
                   else "all-linear")
        report["actions"] = [
            "1. peftkit.plan <model> -> pick the GPU",
            "2. write configs/models/<name>.yaml with the snippet below",
            "3. peftkit.diagnostics --config ... -> fix every FAIL",
            "4. 20-step dry run, then the real run",
        ]
        report["config_snippet"] = {
            "model": {"name_or_path": model, "revision": "<commit-sha>", "max_seq_length": "<p95 tokens>",
                      "trust_remote_code": trust_remote_code},
            "method": {"name": "qlora" if (report.get("params_total") or 0) > 13e9 else "lora",
                       "target_modules": targets},
        }
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Can this model be fine-tuned here, and how?")
    ap.add_argument("model")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true")
    a = ap.parse_args(argv)
    r = check(a.model, a.trust_remote_code, not a.no_tokenizer)
    print(json.dumps(r, indent=2, default=str))
    raise SystemExit(0 if r["verdict"] in (VERDICT_OK, VERDICT_ADV) else 2)


if __name__ == "__main__":
    main()
