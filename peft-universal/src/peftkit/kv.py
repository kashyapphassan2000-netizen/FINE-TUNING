"""KV cache calculator for ANY causal LM: per-token cost, how many users fit, and what each
attention variant would cost. Reads config.json only (no weights).

python -m peftkit.kv meta-llama/Llama-3.1-8B-Instruct --context 32768 --gpu-gb 80 --kv-dtype bf16
"""
from __future__ import annotations

import argparse
import json

from .archs import kv_bytes_per_token, load_config_dict, spec_from_config

KV_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0}


def variants(spec) -> dict:
    """What one token would cost per layer-set under each attention design, for THIS model's shape."""
    hd, L = spec.head_dim, spec.layers
    out = {
        "MHA (kv_heads = heads)": 2 * L * spec.heads * hd,
        f"GQA (kv_heads = {spec.kv_heads})": 2 * L * spec.kv_heads * hd,
        "MQA (kv_heads = 1)": 2 * L * hd,
    }
    if spec.kv_lora_rank:
        out[f"MLA (rank {spec.kv_lora_rank} + rope {spec.qk_rope_head_dim})"] = L * (spec.kv_lora_rank + spec.qk_rope_head_dim)
    return out


def report(model: str, context: int, gpu_gb: float, gpus: int, kv_dtype: str, weight_gb: float | None,
           trust_remote_code: bool = False) -> dict:
    cfg = load_config_dict(model, trust_remote_code)
    spec = spec_from_config(cfg)
    b = KV_BYTES[kv_dtype]
    per_token = kv_bytes_per_token(spec, context, b)
    total = gpu_gb * gpus * 1024**3
    if weight_gb is None:
        from .archs import exact_counts

        c = exact_counts(cfg, trust_remote_code, lora_r=None)
        weight_gb = (c.get("total", 0) * 2) / 1024**3 if c.get("total") else 0.0
    free = total - weight_gb * 1024**3 - 1.5 * 1024**3
    return {
        "model": model, "attention": spec.attention, "layers": spec.layers, "heads": spec.heads,
        "kv_heads": spec.kv_heads, "head_dim": spec.head_dim, "kv_dtype": kv_dtype,
        "kv_bytes_per_token": per_token, "kv_kb_per_token": round(per_token / 1024, 3),
        "gb_per_sequence_at_context": round(per_token * context / 1024**3, 3),
        "weights_gb": round(weight_gb, 2),
        "kv_budget_gb": round(max(0.0, free) / 1024**3, 2),
        "max_cached_tokens": int(max(0.0, free) / per_token) if per_token else 0,
        "max_concurrent_sequences_at_context": int(max(0.0, free) / (per_token * context)) if per_token else 0,
        "variant_bytes_per_token": {k: v * b for k, v in variants(spec).items()},
        "rope": (cfg.get("rope_parameters") or cfg.get("rope_scaling")
                 or {"rope_theta": cfg.get("rope_theta")}),
        "sliding_window": spec.sliding_window, "full_layers": spec.full_layers,
        "sliding_layers": spec.sliding_layers, "linear_layers": spec.linear_layers,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KV cache calculator")
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--gpu-gb", type=float, default=80)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--kv-dtype", default="bf16", choices=list(KV_BYTES))
    ap.add_argument("--weight-gb", type=float, default=None, help="skip weight counting and use this")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args(argv)
    r = report(a.model, a.context, a.gpu_gb, a.gpus, a.kv_dtype, a.weight_gb, a.trust_remote_code)
    print(f"{r['model']}  attention={r['attention']} layers={r['layers']} heads={r['heads']} "
          f"kv_heads={r['kv_heads']} head_dim={r['head_dim']}")
    print(f"KV per token ({r['kv_dtype']}): {r['kv_kb_per_token']} KB   "
          f"one {a.context}-token sequence: {r['gb_per_sequence_at_context']} GB")
    print(f"weights {r['weights_gb']} GB on {a.gpus}x{a.gpu_gb} GB -> KV budget {r['kv_budget_gb']} GB")
    print(f"fits ~{r['max_cached_tokens']:,} cached tokens = "
          f"{r['max_concurrent_sequences_at_context']} sequences at {a.context} tokens")
    print("\nwhat each attention design would cost per token for this shape:")
    for k, v in r["variant_bytes_per_token"].items():
        print(f"  {k:<34}{v / 1024:>9.2f} KB")
    if r["sliding_layers"]:
        print(f"\nsliding window {r['sliding_window']} on {r['sliding_layers']} layers, "
              f"{r['full_layers']} full-attention layers: long contexts cost far less than the formula above")
    print(f"\nrope: {json.dumps(r['rope'])}")


if __name__ == "__main__":
    main()
