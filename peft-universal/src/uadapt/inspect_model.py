"""STEP 0-1: identify and classify an arbitrary model from its module tree, config and forward signature."""
from __future__ import annotations

import inspect as _inspect
import re
from collections import Counter

import torch
from torch import nn

LOSS_NAMES = ["CrossEntropyLoss", "cross_entropy", "BCEWithLogitsLoss", "binary_cross_entropy", "MSELoss",
              "mse_loss", "SmoothL1Loss", "HuberLoss", "l1_loss", "CTCLoss", "ctc_loss", "TripletMarginLoss",
              "CosineEmbeddingLoss", "contrastive", "clip_loss", "KLDivLoss", "kl_div", "NLLLoss", "nll_loss"]


def param_counts(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_type: Counter = Counter()
    for m in model.modules():
        own = sum(p.numel() for p in m.parameters(recurse=False))
        if own:
            by_type[type(m).__name__] += own
    return {"total": total, "trainable": trainable, "by_module_type": dict(by_type.most_common())}


def forward_signature(model: nn.Module) -> dict:
    sig = _inspect.signature(model.forward)
    params = {n: (str(p.default) if p.default is not p.empty else "REQUIRED") for n, p in sig.parameters.items()
              if n not in ("self", "kwargs", "args")}
    return {"parameters": params, "takes_labels": "labels" in params}


def forward_source(model: nn.Module) -> str:
    try:
        return _inspect.getsource(type(model).forward)
    except (OSError, TypeError):
        return ""


def losses_in_source(model: nn.Module) -> list[str]:
    """Scan the forward() source (and the module's file) for loss functions. A hint, not proof."""
    src = forward_source(model)
    try:
        src += _inspect.getsource(_inspect.getmodule(type(model)))
    except (OSError, TypeError):
        pass
    found = []
    for name in LOSS_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", src):
            found.append(name)
    return found


def module_inventory(model: nn.Module) -> dict:
    kinds: Counter = Counter()
    for _, m in model.named_modules():
        t = type(m).__name__
        if isinstance(m, nn.Linear):
            kinds["Linear"] += 1
        elif isinstance(m, nn.Conv1d):
            kinds["Conv1d"] += 1
        elif isinstance(m, nn.Conv2d):
            kinds["Conv2d"] += 1
        elif isinstance(m, nn.Embedding):
            kinds["Embedding"] += 1
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            kinds["Recurrent"] += 1
        elif isinstance(m, nn.MultiheadAttention) or "Attention" in t:
            kinds["Attention"] += 1
        elif "Norm" in t:
            kinds["Norm"] += 1
    return dict(kinds)


def classify(model: nn.Module, config=None) -> dict:
    """STEP 1 decision tree, from evidence. Each conclusion lists the evidence it came from."""
    cfg = config if config is not None else getattr(model, "config", None)
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else (cfg or {})
    inv = module_inventory(model)
    sig = forward_signature(model)["parameters"]
    names = " ".join(n for n, _ in model.named_modules()).lower()
    ev: dict[str, list] = {}

    def add(k, why):
        ev.setdefault(k, []).append(why)

    # modality
    if "pixel_values" in sig or inv.get("Conv2d") or "vision_config" in d or "image_size" in d:
        add("vision", "pixel_values input / Conv2d / vision config")
    if "input_values" in sig or "input_features" in sig or "num_mel_bins" in d or "sampling_rate" in d:
        add("audio", "input_values / input_features / mel bins in config")
    if "input_ids" in sig or inv.get("Embedding"):
        add("text", "input_ids input / token Embedding")
    if ("vision" in ev or "audio" in ev) and "text" in ev and any(k in d for k in ("vision_config", "audio_config", "text_config")):
        add("multimodal", "separate modality sub-configs")
    # structure
    if d.get("is_encoder_decoder") or ("decoder_input_ids" in sig):
        add("encoder-decoder", "is_encoder_decoder / decoder_input_ids")
    elif any(a.endswith("ForCausalLM") for a in d.get("architectures") or []) or type(model).__name__.endswith("ForCausalLM"):
        add("decoder-only", "*ForCausalLM")
    elif "text" in ev and any(k in type(model).__name__ for k in ("ForSequenceClassification", "ForTokenClassification", "ForMaskedLM")) \
            and "decoder" not in names:
        add("encoder-only", "classification / masked-LM head, no decoder")
    if any(k in d for k in ("num_experts", "num_local_experts", "n_routed_experts")) or "experts" in names:
        add("moe", "experts in config or module tree")
    if "state_size" in d or ("x_proj" in names and "dt_proj" in names) or type(model).__name__.lower().startswith("mamba"):
        add("state-space", "state_size / selective-scan projections")
    if inv.get("Recurrent"):
        add("recurrent", "LSTM/GRU/RNN modules")
    if any(k in sig for k in ("timestep", "timesteps", "t", "noise_level")):
        add("diffusion", "timestep argument in forward")
    if (len({k for k in ev if k in ("state-space", "recurrent")}) and inv.get("Attention")):
        add("hybrid", "attention AND recurrent/SSM layers")
    if not hasattr(model, "config"):
        add("custom", "plain nn.Module without a Hugging Face config")
    return {"labels": sorted(ev), "evidence": ev, "inventory": inv}


def inspection_commands() -> str:
    """The commands a human should run, in order, on any unfamiliar model."""
    return """print(model)                                   # module tree
print(getattr(model, "config", None))          # hyper-parameters (HF models)
import inspect; print(inspect.signature(model.forward))   # what forward() expects
print(inspect.getsource(type(model).forward))  # where the loss is computed
for n, m in model.named_modules(): print(n, type(m).__name__)
for n, p in model.named_parameters(): print(n, tuple(p.shape), p.dtype, p.requires_grad)"""
