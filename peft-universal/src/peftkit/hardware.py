"""VRAM estimator for PEFT training and vLLM serving.

These are planning estimates, not guarantees. Real usage depends on kernels,
sequence packing, allocator fragmentation and library versions. Always confirm
with `nvidia-smi` / torch.cuda.max_memory_allocated() on a short dry run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3

# name -> VRAM in GB per GPU (public spec sheets). Prices change: check your provider.
GPU_TABLE: list[tuple[str, int]] = [
    ("RTX 4090 / L4 / A10G", 24),
    ("A100 40GB", 40),
    ("L40 / L40S / RTX A6000", 48),
    ("A100 80GB / H100 80GB", 80),
    ("H100 NVL", 94),
    ("H200", 141),
    ("B200", 180),
]


@dataclass
class Shape:
    vocab: int
    hidden: int
    inter: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    tied: bool = False
    experts: int = 1  # MoE: number of expert MLPs per layer (all are loaded)

    @classmethod
    def from_config(cls, cfg: dict) -> "Shape":
        c = cfg.get("text_config", cfg)  # multimodal configs nest the LM config
        hidden = c["hidden_size"]
        heads = c["num_attention_heads"]
        return cls(
            vocab=c["vocab_size"],
            hidden=hidden,
            inter=c.get("intermediate_size") or c.get("moe_intermediate_size") or 4 * hidden,
            layers=c["num_hidden_layers"],
            heads=heads,
            kv_heads=c.get("num_key_value_heads") or heads,
            head_dim=c.get("head_dim") or hidden // heads,
            tied=bool(c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
            experts=c.get("num_local_experts") or c.get("num_experts") or 1,
        )


def linear_shapes(s: Shape) -> dict[str, tuple[int, int]]:
    """(in_features, out_features) of each linear in one decoder layer (Llama-style)."""
    return {
        "q_proj": (s.hidden, s.heads * s.head_dim),
        "k_proj": (s.hidden, s.kv_heads * s.head_dim),
        "v_proj": (s.hidden, s.kv_heads * s.head_dim),
        "o_proj": (s.heads * s.head_dim, s.hidden),
        "gate_proj": (s.hidden, s.inter),
        "up_proj": (s.hidden, s.inter),
        "down_proj": (s.inter, s.hidden),
    }


def count_params(s: Shape) -> dict[str, int]:
    shp = linear_shapes(s)
    attn = sum(i * o for k, (i, o) in shp.items() if k.endswith(("q_proj", "k_proj", "v_proj", "o_proj")))
    mlp = sum(i * o for k, (i, o) in shp.items() if k in ("gate_proj", "up_proj", "down_proj")) * s.experts
    norms = 2 * s.hidden
    embed = s.vocab * s.hidden
    lm_head = 0 if s.tied else s.vocab * s.hidden
    linear = s.layers * (attn + mlp)
    total = linear + s.layers * norms + s.hidden + embed + lm_head
    return {"linear_in_layers": linear, "embed": embed, "lm_head": lm_head, "total": total}


def adapter_params(s: Shape, r: int, targets: list[str] | str = "all-linear", method: str = "lora") -> int:
    """LoRA adds A (r x in) and B (out x r) per targeted linear: r*(in+out).
    DoRA adds a magnitude vector of size `out` per targeted linear.
    AdaLoRA (SVD form) adds P (r x in), Q (out x r) and diag E (r)."""
    shp = linear_shapes(s)
    names = list(shp) if targets == "all-linear" else list(targets)
    per_layer = 0
    for n in names:
        if n not in shp:
            continue
        i, o = shp[n]
        mult = s.experts if n in ("gate_proj", "up_proj", "down_proj") else 1
        extra = o if method == "dora" else (r if method == "adalora" else 0)
        per_layer += mult * (r * (i + o) + extra)
    return s.layers * per_layer


@dataclass
class TrainEstimate:
    method: str
    base_weights_gb: float
    trainable_params: int
    trainable_state_gb: float
    activations_gb: float
    logits_gb: float
    overhead_gb: float
    total_gb: float


def estimate_train(
    s: Shape,
    method: str = "lora",
    r: int = 16,
    seq_len: int = 2048,
    batch: int = 1,
    grad_ckpt: bool = True,
    optim_8bit: bool = False,
    targets: list[str] | str = "all-linear",
    adalora_init_r: int = 12,
) -> TrainEstimate:
    p = count_params(s)
    nonlinear = p["total"] - p["linear_in_layers"]
    if method == "qlora":
        # NF4 weights ~4 bits + double-quantized absmax constants (~0.127 bit/param, QLoRA paper).
        # prepare_model_for_kbit_training upcasts every non-4-bit tensor (embed, lm_head, norms) to fp32.
        weights = p["linear_in_layers"] * (4.127 / 8) + nonlinear * 4
    else:
        weights = p["total"] * 2  # bf16
    n_train = adapter_params(s, adalora_init_r if method == "adalora" else r, targets, method)
    # adapters are kept in fp32: weight 4 + grad 4 + Adam m,v 8 (8-bit Adam: 2)
    state = n_train * (4 + 4 + (2 if optim_8bit else 8))
    b, L, h, i = batch, s.layers, s.hidden, s.inter
    per_layer_full = seq_len * b * h * (15 + 8 * i / h)  # bf16 bytes, SDPA (no s^2 term)
    if method == "dora":
        per_layer_full *= 1.3  # DoRA keeps extra tensors for the weight-norm path (rough)
    acts = (L * 2 * seq_len * b * h + per_layer_full) if grad_ckpt else L * per_layer_full
    logits = b * seq_len * s.vocab * 4  # one fp32 copy; TRL chunked_nll loss reduces this
    overhead = 1.5 * GIB  # CUDA context + allocator slack
    total = (weights + state + acts + logits) * 1.10 + overhead
    g = lambda x: round(x / GIB, 2)  # noqa: E731
    return TrainEstimate(method, g(weights), n_train, g(state), g(acts), g(logits), g(overhead), g(total))


def estimate_serve(s: Shape, max_len: int = 8192, concurrent: int = 16, weight_bytes: float = 2.0, kv_bytes: float = 2.0) -> dict:
    p = count_params(s)
    w = p["total"] * weight_bytes
    kv_per_token = 2 * s.layers * s.kv_heads * s.head_dim * kv_bytes
    kv = kv_per_token * max_len * concurrent
    total = (w + kv) * 1.10 + 1.5 * GIB
    return {
        "weights_gb": round(w / GIB, 2),
        "kv_cache_per_token_mb": round(kv_per_token / 1024**2, 4),
        "kv_cache_gb": round(kv / GIB, 2),
        "total_gb": round(total / GIB, 2),
        "vllm_gpu_memory_utilization_hint": 0.90,
    }


def recommend(total_gb: float, max_gpus: int = 8) -> list[str]:
    """GPU setups that fit, fewest GPUs first, then smallest card.
    More than one GPU only helps TRAINING memory if you shard (FSDP / DeepSpeed ZeRO-3).
    Plain DDP copies the full model to every GPU. For SERVING, n GPUs = vLLM tensor-parallel n."""
    fits = []
    for n in (1, 2, 4, 8):
        if n > max_gpus:
            break
        for name, vram in GPU_TABLE:
            if vram * n * 0.92 >= total_gb:  # keep ~8% headroom
                fits.append(f"{n}x {name} ({vram * n} GB total)")
                break
    return fits


def load_shape(path_or_id: str) -> Shape:
    p = Path(path_or_id)
    cfg_file = p / "config.json" if p.is_dir() else p
    if cfg_file.is_file():
        return Shape.from_config(json.loads(cfg_file.read_text()))
    from huggingface_hub import hf_hub_download  # only config.json is downloaded, not weights

    return Shape.from_config(json.loads(Path(hf_hub_download(path_or_id, "config.json")).read_text()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Estimate VRAM for PEFT training + vLLM serving")
    ap.add_argument("model", help="HF repo id, local model dir, or path to config.json")
    ap.add_argument("--method", default="all", choices=["all", "lora", "qlora", "dora", "adalora", "loraplus"])
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--serve-max-len", type=int, default=8192)
    ap.add_argument("--serve-concurrency", type=int, default=16)
    a = ap.parse_args(argv)
    s = load_shape(a.model)
    p = count_params(s)
    print(f"shape: {asdict(s)}")
    print(f"params: total={p['total'] / 1e9:.3f}B  linear-in-layers={p['linear_in_layers'] / 1e9:.3f}B")
    methods = ["lora", "qlora", "dora", "adalora", "loraplus"] if a.method == "all" else [a.method]
    print(f"\nTRAINING  seq_len={a.seq_len} batch={a.batch} grad_ckpt={not a.no_grad_ckpt} r={a.r}")
    print(f"{'method':<9}{'weights':>9}{'trainable':>13}{'optim':>8}{'acts':>8}{'logits':>8}{'TOTAL':>8}  fits on")
    for m in methods:
        e = estimate_train(s, m, a.r, a.seq_len, a.batch, not a.no_grad_ckpt, optim_8bit=(m == "qlora"))
        fits = recommend(e.total_gb)
        print(f"{m:<9}{e.base_weights_gb:>9}{e.trainable_params:>13,}{e.trainable_state_gb:>8}"
              f"{e.activations_gb:>8}{e.logits_gb:>8}{e.total_gb:>8}  {fits[0] if fits else 'needs >8 GPUs / offload'}")
    sv = estimate_serve(s, a.serve_max_len, a.serve_concurrency)
    print(f"\nSERVING (vLLM, bf16, max_len={a.serve_max_len}, {a.serve_concurrency} full-length seqs): {sv}")
    fits = recommend(sv["total_gb"])
    print(f"fits on: {fits[0] if fits else 'needs >8 GPUs'}  (tensor-parallel size = GPU count)")
    print("\nMulti-GPU training rows assume FSDP/ZeRO-3 sharding; DDP does not reduce per-GPU memory.")
    print("These are estimates. Confirm with a 20-step dry run and torch.cuda.max_memory_allocated().")


if __name__ == "__main__":
    main()
