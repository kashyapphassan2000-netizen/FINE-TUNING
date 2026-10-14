"""Merge an adapter into a full-precision copy of its base model, and prove the merge is exact.

QLoRA adapters are merged into the bf16 base (NOT the 4-bit one) so the served model has no
double-quantization error. DoRA and AdaLoRA adapters MUST be merged before vLLM can serve them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .peft_factory import DTYPES


def merge(adapter_dir: str, out_dir: str, dtype: str = "bfloat16", base: str | None = None,
          max_shard_size: str = "5GB", check_prompt: str = "Hello, how are you?") -> dict:
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    base = base or cfg["base_model_name_or_path"]
    cuda = torch.cuda.is_available()
    torch_dtype = DTYPES[dtype] if cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype, device_map="auto" if cuda else None)
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    peft_model.eval()
    ids = tok(check_prompt, return_tensors="pt").to(peft_model.device)
    with torch.no_grad():
        before = peft_model(**ids).logits.float()
    merged = peft_model.merge_and_unload(safe_merge=True)
    with torch.no_grad():
        after = merged(**ids).logits.float()
    max_diff = (before - after).abs().max().item()
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size=max_shard_size)
    tok.save_pretrained(out_dir)
    report = {"base": base, "adapter": adapter_dir, "peft_type": cfg.get("peft_type"),
              "use_dora": cfg.get("use_dora", False), "max_abs_logit_diff": max_diff, "out": out_dir}
    (Path(out_dir) / "merge_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Merge PEFT adapter into base model")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--base", default=None, help="override base model id/path")
    ap.add_argument("--max-shard-size", default="5GB")
    a = ap.parse_args(argv)
    r = merge(a.adapter, a.out, a.dtype, a.base, a.max_shard_size)
    print(json.dumps(r))
    tol = 1e-2 if a.dtype != "float32" else 1e-3
    if r["max_abs_logit_diff"] > tol:
        raise SystemExit(f"merge check failed: max diff {r['max_abs_logit_diff']:.2e} > {tol}")


if __name__ == "__main__":
    main()
