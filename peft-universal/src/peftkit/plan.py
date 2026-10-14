"""Universal resource planner: ANY causal LM (millions to trillions of parameters) ->
training memory per method, GPU count per GPU type, multi-node flag, serving VRAM + KV cache capacity,
compute (FLOPs), wall-clock time and cost.

python -m peftkit.plan <hub-id | dir | config.json> --method all --seq-len 4096 \
    --dataset-tokens 5e7 --epochs 2 --tokens-per-sec 3000 --price-per-gpu-hour <your price> \
    --serve-dtype fp8 --serve-len 32768 --concurrency 64

    --dataset data/processed/train.jsonl        # counts the tokens for you with the model's tokenizer

Methods: lora qlora dora adalora loraplus, and `full` (full fine-tuning: every weight trained, ZeRO-3/FSDP).
Also printed: host RAM and disk to provision, and training hours per GPU type (from MFU x peak FLOPs when you
have not measured tokens/sec yet).
Every number is an ESTIMATE. Confirm memory with a 20-step dry run and throughput with a measurement.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

from .archs import exact_counts, kv_bytes_per_token, load_config_dict, spec_from_config

GIB = 1024**3
GPU_VRAM_GB = {  # public spec sheets; prices are NOT included on purpose
    "RTX4090-24": 24, "L4-24": 24, "A10G-24": 24, "A100-40": 40, "L40S-48": 48, "RTX6000Ada-48": 48,
    "A100-80": 80, "H100-80": 80, "H100NVL-94": 94, "H200-141": 141, "B200-180": 180,
}
BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "mxfp4": 4.25 / 8, "int4": 0.5,
         "nf4": 4.127 / 8}
HEADROOM = 0.90       # never plan above 90% of VRAM
GPUS_PER_NODE = 8
# Dense bf16 tensor-core peak per GPU, FLOP/s, from NVIDIA spec sheets (WITHOUT 2:4 sparsity, which training
# does not use). RTX 4090: the 165 TFLOPs figure is with fp16 accumulate; bf16 with fp32 accumulate is lower.
GPU_PEAK_BF16 = {
    "RTX4090-24": 165e12, "L4-24": 121e12, "A100-40": 312e12, "L40S-48": 362e12, "RTX6000Ada-48": 364e12,
    "A100-80": 312e12, "H100-80": 989e12, "H100NVL-94": 835e12, "H200-141": 989e12, "B200-180": 2250e12,
}
# Typical model-FLOPs utilisation for fine-tuning runs (single node, packing off). Measure yours: these only
# get you to the right order of magnitude before the dry run. QLoRA pays for 4-bit dequantisation.
DEFAULT_MFU = {"full": 0.40, "lora": 0.35, "dora": 0.25, "adalora": 0.30, "loraplus": 0.35, "qlora": 0.25}


def train_memory(counts: dict, trainable: int, spec, method: str, seq: int, batch: int,
                 grad_ckpt: bool = True, upcast: bool = True) -> dict:
    total, active, embed = counts["total"], counts["active"], counts["embed_and_head"]
    if method == "qlora":
        # bitsandbytes quantizes nn.Linear only. Fused MoE expert tensors stay unquantized, and PEFT's
        # prepare_model_for_kbit_training upcasts every non-4-bit tensor to fp32 (verified in source + test).
        lin = counts.get("linear_params", total - embed)
        fused = counts.get("fused_params", 0)
        other = total - lin - fused
        up = 4.0 if upcast else 2.0
        weights = lin * BYTES["nf4"] + fused * up + other * up
        optim = trainable * (4 + 4 + 2)                  # paged 8-bit Adam
    elif method == "full":
        # mixed-precision AdamW: bf16 weights (2) + bf16 grads (2) + fp32 master (4) + Adam m, v (8) = 16 B/param
        weights = total * 2.0
        optim = total * (2 + 4 + 8)
    else:
        weights = total * 2.0
        optim = trainable * (4 + 4 + 8)
    h, L = spec.hidden, spec.layers
    # activations scale with ACTIVE width; for MoE use active-per-layer as a proxy for the MLP part
    mlp_ratio = max(1.0, (active - embed) / max(1, L) / (12 * h * h)) * 4
    per_layer = seq * batch * h * (15 + 2 * mlp_ratio)
    if method == "dora":
        per_layer *= 1.3
    acts = L * 2 * seq * batch * h + per_layer if grad_ckpt else L * per_layer
    logits = batch * seq * spec.vocab * 4
    total_b = (weights + optim + acts + logits) * 1.10 + 1.5 * GIB
    if method == "full":
        trainable = total
    return {"method": method, "upcast_non_quantized": upcast if method == "qlora" else None,
            "weights_gb": weights / GIB, "optim_gb": optim / GIB, "activations_gb": acts / GIB,
            "logits_gb": logits / GIB, "total_gb": total_b / GIB, "trainable": trainable}


def gpus_needed(total_gb: float, vram_gb: float) -> int:
    return max(1, math.ceil(total_gb / (vram_gb * HEADROOM)))


def layout(n: int) -> str:
    if n <= 1:
        return "single GPU"
    if n <= GPUS_PER_NODE:
        return f"1 node, {n} GPUs (needs sharding: FSDP/ZeRO-3 for training, tensor parallel for serving)"
    nodes = math.ceil(n / GPUS_PER_NODE)
    return (f"{nodes} nodes x {GPUS_PER_NODE} GPUs (multi-node: fast interconnect such as InfiniBand required; "
            "use expert/pipeline parallel frameworks)")


def compute_flops(active: int, tokens: float, grad_ckpt: bool = True, full: bool = False) -> float:
    """PEFT with a frozen base: forward 2N + activation-gradient backward 2N (+2N recompute with checkpointing)
    = 6N per token. FULL fine-tuning also computes weight gradients (+2N): 6N, or 8N with checkpointing."""
    if full:
        return (8.0 if grad_ckpt else 6.0) * active * tokens
    return (6.0 if grad_ckpt else 4.0) * active * tokens


def host_requirements(total: int, method: str, save_total_limit: int = 3, shard_gb: float = 5.0) -> dict:
    """CPU RAM and disk of the TRAINING machine (VRAM is the GPU table). Rules, not measurements:
    - disk: the downloaded checkpoint (bf16) + one merged copy + checkpoints + ~40 GB for envs/caches.
      Full FT checkpoints hold optimizer state too (~16 B/param each); LoRA checkpoints are megabytes.
    - RAM: transformers streams shards straight to the GPU, so loading needs ~2 shards; but MERGING on the CPU,
      FSDP rank-0 loading (cpu_ram_efficient_loading) and saving a full model need the WHOLE model in RAM."""
    w = total * 2.0 / GIB
    ckpt = (total * 16.0 / GIB) if method == "full" else 0.5
    disk = w + w + ckpt * save_total_limit + 40
    return {"bf16_checkpoint_gb": round(w, 1),
            "disk_gb": round(disk, 1),
            "ram_min_gb": round(max(16.0, 2 * shard_gb + 8), 1),
            "ram_recommended_gb": round(max(32.0, 1.25 * w + 16), 1),
            "explain": "disk = download + merged copy + checkpoints x save_total_limit + 40 GB; "
                       "RAM recommended = 1.25 x bf16 weights + 16 GB (CPU merge / FSDP rank-0 load)"}


def time_per_gpu_type(active: int, tokens: float, method: str, gpus: dict, mfu: float | None = None,
                      price: float | dict | None = None) -> dict:
    """hours = k x active x tokens / (n_gpus x peak x MFU), k = 6 (PEFT) or 8 (full FT, checkpointed).
    Assumes near-linear scaling inside a node; multi-node adds communication overhead (budget 10-30% more)."""
    m = mfu or DEFAULT_MFU.get(method, 0.3)
    flops = compute_flops(active, tokens, full=(method == "full"))
    out = {}
    for g, peak in GPU_PEAK_BF16.items():
        n = gpus.get(g, {}).get("n", 1)
        gpu_hours = round(flops / (peak * m) / 3600, 2)    # cost below = these printed hours x price, checkable
        row = {"gpus": n, "mfu": m, "gpu_hours": gpu_hours, "wall_clock_hours": round(gpu_hours / n, 2)}
        pr = price.get(g) if isinstance(price, dict) else price   # per-GPU-type price, or one price for all
        if pr:
            row["cost"] = round(gpu_hours * pr, 2)
            row["price_per_gpu_hour"] = pr
        out[g] = row
    return out


def count_dataset_tokens(path: str, tokenizer_id: str, trust_remote_code: bool = False) -> dict:
    """Exact tokens of a chat JSONL with the model's own tokenizer and chat template (chars/4 fallback)."""
    import json as _json
    from pathlib import Path as _P

    rows = [_json.loads(x) for x in _P(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    except Exception:
        pass
    lens = []
    for r in rows:
        msgs = r.get("messages") or [{"role": "user", "content": r.get("text", _json.dumps(r))}]
        if tok is not None and getattr(tok, "chat_template", None):
            lens.append(len(tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"]))
        else:
            lens.append(sum(len(str(m.get("content", ""))) for m in msgs) // 4)
    lens.sort()
    return {"rows": len(rows), "tokens": sum(lens), "p95": lens[int(0.95 * (len(lens) - 1))] if lens else 0,
            "max": lens[-1] if lens else 0, "exact": tok is not None}


def serving(total: int, spec, dtype: str, seq_len: int, concurrency: int, kv_dtype: str = "bf16") -> dict:
    w = total * BYTES[dtype]
    kv_tok = kv_bytes_per_token(spec, seq_len, BYTES[kv_dtype])
    kv = kv_tok * seq_len * concurrency
    need = (w + kv) / HEADROOM + 1.5 * GIB
    out = {"weights_dtype": dtype, "weights_gb": w / GIB, "kv_mb_per_token": kv_tok / 1024**2,
           "kv_gb_for_load": kv / GIB, "total_gb": need / GIB, "per_gpu_type": {}}
    for g, v in GPU_VRAM_GB.items():
        n = gpus_needed(need / GIB, v)
        n = 2 ** math.ceil(math.log2(n)) if n > 1 else 1  # tensor-parallel sizes are usually powers of two
        free = n * v * HEADROOM * GIB - w - 1.5 * GIB
        out["per_gpu_type"][g] = {"gpus": n, "max_concurrent_tokens": int(max(0, free) / kv_tok) if kv_tok else None,
                                  "layout": layout(n)}
    return out


def plan(model: str, methods: list[str], seq: int, batch: int, r: int, dataset_tokens: float | None,
         epochs: float, tps: float | None, price: float | None, serve_dtype: str, serve_len: int,
         concurrency: int, trust_remote_code: bool = False, targets: str = "all-linear",
         upcast: bool = True, gpu_flops: float | None = None, mfu: float | None = None,
         dataset: str | None = None, gpu_prices: dict | None = None, tokenizer: str | None = None) -> dict:
    cfg = load_config_dict(model, trust_remote_code)
    spec = spec_from_config(cfg)
    counts = exact_counts(cfg, trust_remote_code, lora_r=r)
    if not counts or "total" not in counts:
        from .hardware import Shape, adapter_params, count_params

        sh = Shape.from_config(cfg)
        p = count_params(sh)
        counts = {"total": p["total"], "active": p["total"], "embed_and_head": p["embed"] + p["lm_head"],
                  "lora_trainable_all_linear": adapter_params(sh, r), "source": "formula fallback (dense Llama-style)",
                  **({"meta_error": counts.get("error")} if counts else {})}
    else:
        counts["source"] = "exact (meta-device instantiation)"
    key = "lora_trainable_attention" if targets == "attention" else "lora_trainable_all_linear"
    trainable = counts.get(key) or counts.get("lora_trainable_all_linear") or 0
    result = {"model": model, "spec": asdict(spec), "counts": counts, "training": [], "serving": None}
    if dataset:
        ds = count_dataset_tokens(dataset, tokenizer or model, trust_remote_code)
        result["dataset"] = ds
        dataset_tokens = dataset_tokens or ds["tokens"]
    for m in methods:
        tr = train_memory(counts, int(trainable * (1.03 if m == "dora" else 1.0)), spec, m, seq, batch,
                          upcast=upcast)
        tr["targets"] = targets
        tr["gpus"] = {g: {"n": gpus_needed(tr["total_gb"], v), "layout": layout(gpus_needed(tr["total_gb"], v))}
                      for g, v in GPU_VRAM_GB.items()}
        result["training"].append(tr)
    result["host"] = host_requirements(counts["total"], methods[0] if len(methods) == 1 else "lora")
    if any(m == "full" for m in methods):
        result["host_full"] = host_requirements(counts["total"], "full")
    if dataset_tokens:
        tokens = dataset_tokens * epochs
        result["time"] = {t["method"]: time_per_gpu_type(counts["active"], tokens, t["method"], t["gpus"], mfu,
                                                         gpu_prices or price)
                          for t in result["training"]}
        flops = compute_flops(counts["active"], tokens)
        comp = {"tokens_processed": tokens, "flops": flops, "note": "FLOPs use ACTIVE parameters"}
        if not tps and gpu_flops and mfu:
            tps = mfu * gpu_flops / ((flops / tokens))
            comp["tokens_per_sec_derived"] = tps
            comp["derived_from"] = "tokens/sec = MFU x GPU peak FLOPs/sec / (6 x active params)"
        if tps:
            comp["gpu_hours_per_gpu_rate"] = tokens / tps / 3600
            comp["explain"] = "tokens / (measured tokens/sec per GPU) / 3600 = GPU-hours; wall-clock = GPU-hours / GPUs"
            if price:
                comp["cost"] = comp["gpu_hours_per_gpu_rate"] * price
        result["compute"] = comp
    result["serving"] = serving(counts["total"], spec, serve_dtype, serve_len, concurrency)
    return result


def _fmt(n: float) -> str:
    return f"{n / 1e9:,.2f}B" if n >= 1e9 else f"{n / 1e6:,.1f}M"


def print_plan(p: dict) -> None:
    s, c = p["spec"], p["counts"]
    print(f"MODEL {p['model']}  type={s['model_type']}  family={s['family']}  attention={s['attention']}")
    print(f"  layers={s['layers']} hidden={s['hidden']} heads={s['heads']} kv_heads={s['kv_heads']} "
          f"vocab={s['vocab']} experts={s['experts']} top-k={s['topk']} shared={s['shared']}")
    print(f"  params total={_fmt(c['total'])} active/token={_fmt(c['active'])} [{c['source']}]")
    if c.get("linear_names"):
        print(f"  LoRA target names: {c['linear_names']}  fused expert tensors: {c.get('fused_expert_params')}")
    for n in s["notes"]:
        print(f"  NOTE: {n}")
    print("\nTRAINING (per replica, estimates)")
    for t in p["training"]:
        best = sorted(t["gpus"].items(), key=lambda kv: (kv[1]["n"], GPU_VRAM_GB[kv[0]]))[:3]
        opts = "; ".join(f"{v['n']}x {g}" for g, v in best)
        opts += f"  [{best[0][1]['layout'].split(' (')[0]}]"
        print(f"  {t['method']:<9} weights {t['weights_gb']:9.1f} GB  optim {t['optim_gb']:6.1f}  acts "
              f"{t['activations_gb']:6.1f}  logits {t['logits_gb']:5.1f}  TOTAL {t['total_gb']:9.1f} GB  -> {opts}")
    print(f"  trainable LoRA params ({p['training'][0]['targets']}): {p['training'][0]['trainable']:,}"
          f"   all-linear={c.get('lora_trainable_all_linear')}  attention-only={c.get('lora_trainable_attention')}")
    if p.get("compute"):
        cp = p["compute"]
        print(f"\nCOMPUTE tokens={cp['tokens_processed']:.3g}  FLOPs={cp['flops']:.3g}")
        if "gpu_hours_per_gpu_rate" in cp:
            print(f"  GPU-hours = {cp['gpu_hours_per_gpu_rate']:,.1f}  (wall-clock = GPU-hours / number of GPUs)")
        if "cost" in cp:
            print(f"  cost = {cp['cost']:,.2f} in your price's currency")
    if p.get("dataset"):
        d = p["dataset"]
        print(f"\nDATASET {d['rows']:,} rows, {d['tokens']:,} tokens ({'exact' if d['exact'] else 'chars/4 estimate'}), "
              f"p95 {d['p95']} tokens -> set model.max_seq_length >= {d['p95']}")
    if p.get("time"):
        print("\nTIME for all epochs: hours = k x active x tokens / (GPUs x peak bf16 x MFU), k=6 PEFT, 8 full FT."
              "  MEASURE tokens/sec in the dry run")
        for meth, rows in p["time"].items():
            cells = []
            for g in ("RTX4090-24", "L40S-48", "A100-80", "H100-80", "H200-141", "B200-180"):
                r = rows[g]
                cells.append(f"{g.split('-')[0]} {r['gpus']}x {r['wall_clock_hours']:.1f}h"
                             + (f" ${r['cost']:,.0f}" if "cost" in r else ""))
            print(f"  {meth:<9}" + " | ".join(cells))
    h = p.get("host")
    if h:
        print(f"\nHOST (training VM): disk >= {h['disk_gb']:.0f} GB, RAM min {h['ram_min_gb']:.0f} GB, "
              f"recommended {h['ram_recommended_gb']:.0f} GB  [{h['explain']}]")
        if p.get("host_full"):
            hf = p["host_full"]
            print(f"  full fine-tuning checkpoints: disk >= {hf['disk_gb']:.0f} GB (optimizer state in every checkpoint)")
    sv = p["serving"]
    print(f"\nSERVING weights {sv['weights_dtype']}={sv['weights_gb']:.1f} GB  KV {sv['kv_mb_per_token']:.4f} MB/token"
          f"  KV for your load={sv['kv_gb_for_load']:.1f} GB  total={sv['total_gb']:.1f} GB")
    for g, v in sv["per_gpu_type"].items():
        print(f"  {g:<14} {v['gpus']:>4} GPUs  max concurrent tokens ~{v['max_concurrent_tokens']:,}  {v['layout']}")
    print("\nConfirm: 20-step dry run (peak_vram_gb in run_meta.json) and measured tokens/sec before buying hours.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Universal GPU/compute/cost planner for any causal LM")
    ap.add_argument("model")
    ap.add_argument("--method", default="all", choices=["all", "lora", "qlora", "dora", "adalora", "loraplus", "full"])
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--dataset-tokens", type=float, default=None, help="total training tokens per epoch")
    ap.add_argument("--dataset", default=None, help="train.jsonl: counts tokens with the model's tokenizer")
    ap.add_argument("--tokenizer", default=None, help="tokenizer for --dataset when MODEL is only a config.json")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--tokens-per-sec", type=float, default=None, help="MEASURED per GPU in a dry run")
    ap.add_argument("--price-per-gpu-hour", type=float, default=None, help="one price used for EVERY GPU type")
    ap.add_argument("--gpu-price", nargs="*", default=[], metavar="GPU=PRICE",
                    help="per-type prices you were quoted, e.g. H100-80=2.39 RTX4090-24=0.35 A100-80=1.6")
    ap.add_argument("--serve-dtype", default="bf16", choices=list(BYTES))
    ap.add_argument("--serve-len", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--targets", default="all-linear", choices=["all-linear", "attention"],
                    help="attention = do not put LoRA on MoE experts / MLPs")
    ap.add_argument("--no-upcast", action="store_true", help="QLoRA prep keeps non-4-bit tensors in bf16")
    ap.add_argument("--gpu-flops", type=float, default=None, help="GPU peak FLOPs/sec from its spec (optional)")
    ap.add_argument("--mfu", type=float, default=None, help="model FLOPs utilisation you expect, e.g. 0.3")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    methods = ["lora", "qlora", "dora", "loraplus", "full"] if a.method == "all" else [a.method]
    p = plan(a.model, methods, a.seq_len, a.batch, a.r, a.dataset_tokens, a.epochs, a.tokens_per_sec,
             a.price_per_gpu_hour, a.serve_dtype, a.serve_len, a.concurrency, a.trust_remote_code,
             a.targets, not a.no_upcast, a.gpu_flops, a.mfu, a.dataset,
             {k: float(v) for k, v in (x.split("=", 1) for x in a.gpu_price)} or None, a.tokenizer)
    if a.json:
        print(json.dumps(p, indent=2, default=str))
    else:
        print_plan(p)


if __name__ == "__main__":
    main()
