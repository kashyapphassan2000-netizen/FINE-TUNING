"""Decide HOW an adapter can be served by vLLM 0.29.0 (rules read from vllm/lora/peft_helper.py
and vllm/config/lora.py at tag v0.29.0).

- plain LoRA (lora / qlora / loraplus): serve as a hot-swappable adapter with --enable-lora
- DoRA: vLLM rejects it ("vLLM does not yet support DoRA")  -> merge first
- AdaLoRA: peft_type ADALORA, not LoRA format                 -> merge first
- modules_to_save set: vLLM rejects it                         -> merge first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)  # vllm 0.29.0 MaxLoRARanks


def check(adapter_dir: str) -> dict:
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    reasons = []
    if cfg.get("peft_type") != "LORA":
        reasons.append(f"peft_type={cfg.get('peft_type')} (vLLM loads LoRA-format adapters only)")
    if cfg.get("use_dora"):
        reasons.append("DoRA adapters are not supported by vLLM 0.29.0")
    if cfg.get("modules_to_save"):
        reasons.append("modules_to_save must be empty for vLLM")
    if cfg.get("bias", "none") != "none":
        reasons.append("bias must be 'none'")
    ranks = [cfg.get("r", 0)] + [int(v) for v in (cfg.get("rank_pattern") or {}).values() if isinstance(v, int)]
    need = max(ranks)
    max_rank = next((x for x in VLLM_MAX_LORA_RANKS if x >= need), None)
    if max_rank is None:
        reasons.append(f"rank {need} exceeds vLLM maximum 512")
    ok = not reasons
    return {
        "adapter": adapter_dir,
        "servable_as_adapter": ok,
        "reasons": reasons,
        "max_lora_rank_flag": max_rank,
        "base_model": cfg.get("base_model_name_or_path"),
        "next_step": ("serve with: --enable-lora --max-lora-rank "
                      f"{max_rank} --lora-modules <name>={adapter_dir}") if ok else
                     f"python -m peftkit.merge --adapter {adapter_dir} --out <merged_dir>  then serve the merged dir",
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    r = check(ap.parse_args(argv).adapter)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
