"""Universal architecture analysis for ANY Hugging Face causal LM, from config.json only.

Why: architectures differ (dense GQA, MoE, MLA, sliding-window mixes, hybrid linear-attention/SSM).
Hand formulas break on each new family. Instead we build the model on the `meta` device (no memory,
no weights) with the real transformers code and count exactly, then read the attention layout for the
KV cache. Validated: DeepSeek-V3 671.0B, Llama-3.1-405B 405.9B, Qwen3-235B-A22B 235.1B/22.2B active,
Kimi-K2 1026.4B/32.9B active, Mixtral 46.7B (see tests and the guide).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Different families name the same thing differently. First match wins.
ALIASES = {
    "hidden": ["hidden_size", "d_model", "n_embd", "dim"],
    "layers": ["num_hidden_layers", "n_layer", "num_layers", "n_layers"],
    "heads": ["num_attention_heads", "n_head", "num_heads", "n_heads"],
    "kv_heads": ["num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num"],
    "vocab": ["vocab_size", "padded_vocab_size"],
    "experts": ["n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts", "n_experts"],
    "topk": ["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k", "n_activated_experts"],
    "shared": ["n_shared_experts", "num_shared_experts", "shared_expert_count"],
    "max_pos": ["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"],
}


def pick(cfg: dict, key: str, default=None):
    for k in ALIASES[key]:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def text_config(cfg: dict) -> dict:
    """Multimodal repos nest the language model config."""
    for k in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(k), dict):
            merged = dict(cfg[k])
            merged.setdefault("model_type", cfg[k].get("model_type", cfg.get("model_type")))
            return merged
    return cfg


@dataclass
class ArchSpec:
    model_type: str
    architectures: list
    family: str               # dense | moe
    attention: str            # gqa | mha | mqa | mla | hybrid-linear | sliding-mix
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    max_pos: int | None
    experts: int = 0
    topk: int = 0
    shared: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    sliding_window: int | None = None
    full_layers: int = 0
    sliding_layers: int = 0
    linear_layers: int = 0    # linear-attention / SSM layers: constant-size state, no growing KV
    tied: bool = False
    quantization: str | None = None
    notes: list = field(default_factory=list)


def spec_from_config(cfg: dict) -> ArchSpec:
    raw = cfg
    cfg = text_config(cfg)
    layers = int(pick(cfg, "layers"))
    hidden = int(pick(cfg, "hidden"))
    heads = int(pick(cfg, "heads", 1))
    kv = int(pick(cfg, "kv_heads", heads) or heads)
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    experts = int(pick(cfg, "experts", 0) or 0)
    lt = cfg.get("layer_types") or []
    full = sum(t in ("full_attention", "attention", "chunked_attention") for t in lt) if lt else layers
    sliding = sum(t == "sliding_attention" for t in lt)
    linear = sum(t in ("linear_attention", "mamba", "mamba2", "ssm", "conv") for t in lt)
    if lt and full + sliding + linear != layers:
        full = layers - sliding - linear  # unknown labels: assume full attention (conservative)
    if cfg.get("kv_lora_rank"):
        att = "mla"
    elif linear:
        att = "hybrid-linear"
    elif sliding:
        att = "sliding-mix"
    elif kv == 1:
        att = "mqa"
    elif kv < heads:
        att = "gqa"
    else:
        att = "mha"
    q = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    s = ArchSpec(
        model_type=cfg.get("model_type", raw.get("model_type", "unknown")),
        architectures=raw.get("architectures") or [],
        family="moe" if experts > 1 else "dense",
        attention=att, layers=layers, hidden=hidden, heads=heads, kv_heads=kv, head_dim=head_dim,
        vocab=int(pick(cfg, "vocab")), max_pos=pick(cfg, "max_pos"), experts=experts,
        topk=int(pick(cfg, "topk", 0) or 0), shared=int(pick(cfg, "shared", 0) or 0),
        kv_lora_rank=int(cfg.get("kv_lora_rank") or 0), qk_rope_head_dim=int(cfg.get("qk_rope_head_dim") or 0),
        sliding_window=cfg.get("sliding_window"), full_layers=full, sliding_layers=sliding, linear_layers=linear,
        tied=bool(cfg.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))),
        quantization=q.get("quant_method") if isinstance(q, dict) else None,
    )
    if s.quantization:
        s.notes.append(f"checkpoint is pre-quantized ({s.quantization}); PEFT training usually needs a bf16 copy "
                       "(dequantize first) unless your stack supports training on this format")
    if s.family == "moe":
        s.notes.append("MoE: ALL experts must be in GPU memory; compute per token uses only the active experts")
    if s.linear_layers:
        s.notes.append("hybrid: linear-attention/SSM layers keep a constant-size state instead of a growing KV cache")
    return s


def kv_bytes_per_token(s: ArchSpec, seq_len: int, dtype_bytes: float = 2.0) -> float:
    """Average KV-cache bytes per token for a sequence of length seq_len (per sequence, all layers)."""
    if s.attention == "mla":
        # MLA caches one compressed latent (kv_lora_rank) + the decoupled RoPE key per layer
        return s.layers * (s.kv_lora_rank + s.qk_rope_head_dim) * dtype_bytes
    per_layer = 2 * s.kv_heads * s.head_dim * dtype_bytes
    total = s.full_layers * per_layer
    if s.sliding_layers and s.sliding_window:
        total += s.sliding_layers * per_layer * min(1.0, s.sliding_window / max(seq_len, 1))
    elif s.sliding_layers:
        total += s.sliding_layers * per_layer
    return total


def load_config_dict(model: str, trust_remote_code: bool = False) -> dict:
    p = Path(model)
    f = p / "config.json" if p.is_dir() else p
    if f.is_file():
        return json.loads(f.read_text())
    from huggingface_hub import hf_hub_download  # downloads config.json only

    return json.loads(Path(hf_hub_download(model, "config.json")).read_text())


_CACHE: dict = {}


def exact_counts(cfg: dict, trust_remote_code: bool = False, lora_r: int | None = 16) -> dict:
    """Cached wrapper: giant MoE configs take a while to build even on the meta device."""
    key = (json.dumps(cfg, sort_keys=True, default=str), trust_remote_code, lora_r)
    if key not in _CACHE:
        _CACHE[key] = _exact_counts(cfg, trust_remote_code, lora_r)
    return dict(_CACHE[key])


def _exact_counts(cfg: dict, trust_remote_code: bool = False, lora_r: int | None = 16) -> dict:
    """Instantiate on the meta device with the real modeling code and count parameters exactly.
    Returns {} if the architecture cannot be built locally (then use the formula estimator)."""
    import warnings

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="peft")
    tc = text_config(cfg)
    try:
        conf_kwargs = {k: v for k, v in tc.items() if k not in ("model_type", "transformers_version", "architectures")}
        conf = AutoConfig.for_model(tc["model_type"], **conf_kwargs)
    except Exception:
        return {}
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(conf, trust_remote_code=trust_remote_code)
    except Exception as exc:  # custom code, missing deps, multimodal wrappers...
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    total = expert = embed = fused_n = linear_n = 0
    fused, linear, attn = set(), set(), set()
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if ".experts." in name or name.endswith(".experts"):
            expert += n
        if p.dim() == 3:  # fused expert tensors: NOT nn.Linear, so bitsandbytes will not quantize them
            fused.add(name.split(".")[-1])
            fused_n += n
        if "embed_tokens" in name or name.startswith("lm_head") or ".lm_head" in name:
            embed += n
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "Linear" and "lm_head" not in name:
            leaf = name.split(".")[-1]
            linear.add(leaf)
            linear_n += sum(q.numel() for q in mod.parameters(recurse=False))
            if "attn" in name or "attention" in name:
                attn.add(leaf)
    s = spec_from_config(cfg)
    active = total - expert + (expert * s.topk / s.experts if s.experts and s.topk else 0)
    out = {"total": total, "active": int(active), "expert": expert, "embed_and_head": embed,
           "linear_params": linear_n, "fused_params": fused_n,
           "other_params": total - linear_n - fused_n,
           "linear_names": sorted(linear), "attention_linear_names": sorted(attn),
           "fused_expert_params": sorted(fused)}
    if lora_r:
        from peft import LoraConfig, get_peft_model

        for key, targets in (("lora_trainable_all_linear", "all-linear"),
                             ("lora_trainable_attention", sorted(attn) or None)):
            if targets is None:
                continue
            try:
                with torch.device("meta"):
                    fresh = AutoModelForCausalLM.from_config(conf, trust_remote_code=trust_remote_code)
                pm = get_peft_model(fresh, LoraConfig(r=lora_r, target_modules=targets))
                out[key] = pm.get_nb_trainable_parameters()[0]
            except Exception as exc:
                out[key.replace("trainable", "error")] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def describe(model: str, trust_remote_code: bool = False) -> dict:
    cfg = load_config_dict(model, trust_remote_code)
    s = spec_from_config(cfg)
    return {"spec": asdict(s), "counts": exact_counts(cfg, trust_remote_code)}
