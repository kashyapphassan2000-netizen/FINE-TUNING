"""UNIVERSAL LOSS AND CONTRACT DERIVATION.

Given a model and one batch, derive from FIRST PRINCIPLES:
  input contract  -> what forward() accepts, with shapes and dtypes actually observed
  output contract -> what it returns, and what those numbers MEAN (vocab logits? class logits? values?)
  loss            -> the objective implied by that contract, with the rationale and a confidence
This is the step the review called "universal loss derivation": it is rules over contracts, not a lookup
table of model names.
"""
from __future__ import annotations

import inspect as _inspect

import torch
from torch import nn

from .tasks import TASKS, contrastive_loss, diffusion_loss, hf_loss, regression_loss, reward_model_loss


def input_contract(model, batch) -> dict:
    sig = _inspect.signature(model.forward).parameters
    got = {}
    b = batch if isinstance(batch, dict) else {f"arg{i}": v for i, v in enumerate(batch)}
    for k, v in b.items():
        got[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype)} if torch.is_tensor(v) else type(v).__name__
    return {"accepts": [k for k in sig if k != "kwargs"], "provided": got,
            "missing_required": [k for k, p in sig.items()
                                 if p.default is p.empty and k not in ("self", "args", "kwargs") and k not in b]}


def output_contract(model, batch) -> dict:
    with torch.no_grad():
        out = model(**batch) if isinstance(batch, dict) else model(*batch)
    info = {"type": type(out).__name__, "has_loss": getattr(out, "loss", None) is not None, "tensors": {}}
    if hasattr(out, "items"):
        items = list(out.items())
    elif torch.is_tensor(out):
        items = [("output", out)]
    elif isinstance(out, (tuple, list)):
        items = [(f"output_{i}", v) for i, v in enumerate(out)]
    else:
        items = []
    for k, v in items:
        if torch.is_tensor(v):
            info["tensors"][k] = {"shape": tuple(v.shape), "dtype": str(v.dtype), "floating": v.is_floating_point()}
    cfg = getattr(model, "config", None)
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
    # num_labels is a PROPERTY on HF configs (derived from id2label) and is missing from to_dict()
    if cfg is not None and getattr(cfg, "num_labels", None):
        d.setdefault("num_labels", cfg.num_labels)
    main = info["tensors"].get("logits") or info["tensors"].get("output") or next(iter(info["tensors"].values()), None)
    if main:
        last = main["shape"][-1]
        info["last_dim"] = last
        # width 1 is checked FIRST: a "1-class classifier" is a value (regression or a binary logit),
        # never a class distribution. Found by the tests.
        info["last_dim_means"] = ("single value" if last == 1 else
                                  "vocabulary" if last == d.get("vocab_size") else
                                  "classes" if last == d.get("num_labels") else
                                  "hidden state / embedding" if last in (d.get("hidden_size"), d.get("d_model")) else
                                  "unknown - decide yourself")
        info["rank"] = len(main["shape"])
    return info


def derive_loss(model, batch, pairs: bool = False, timestep: bool = False) -> dict:
    """Rules over contracts. Every branch states WHY, and what would make it wrong."""
    ic, oc = input_contract(model, batch), output_contract(model, batch)
    sig = ic["accepts"]
    has_labels = "labels" in (batch if isinstance(batch, dict) else {})
    if oc["has_loss"] and has_labels:
        return {"loss_fn": hf_loss, "name": "model's own loss", "confidence": "high",
                "why": "the model returned a loss for a batch containing labels",
                "wrong_if": "you need a different objective than the head implements"}
    if timestep or any(k in sig for k in ("timestep", "timesteps", "noise_level")):
        return {"loss_fn": diffusion_loss, "name": "diffusion epsilon-MSE", "confidence": "medium",
                "why": "forward() takes a timestep and returns a tensor shaped like its input: a denoiser",
                "wrong_if": "the model predicts v or x0 instead of epsilon - check its training code"}
    if pairs:
        if oc.get("last_dim") == 1:
            return {"loss_fn": reward_model_loss, "name": "Bradley-Terry pairwise", "confidence": "high",
                    "why": "paired inputs and a scalar output: a preference scorer",
                    "wrong_if": "you wanted DPO (that trains a policy, not a scorer)"}
        return {"loss_fn": contrastive_loss, "name": "multiple-negatives contrastive", "confidence": "high",
                "why": "paired inputs and embedding outputs: similarity learning with in-batch negatives",
                "wrong_if": "your positives are not truly related, or the batch is too small for negatives"}
    means, rank = oc.get("last_dim_means"), oc.get("rank")
    if means == "vocabulary" and rank == 3:
        return {"loss_fn": hf_loss, "name": "token cross-entropy (shifted for causal LMs)", "confidence": "high",
                "why": "3-D logits over the vocabulary: next-token or masked-token prediction",
                "wrong_if": "labels are not aligned/shifted the way the head expects"}
    if means == "classes" and rank == 2:
        return {"loss_fn": hf_loss, "name": "cross-entropy over classes", "confidence": "high",
                "why": "2-D logits with width = num_labels", "wrong_if": "the task is multi-label (use BCE)"}
    if means == "classes" and rank == 3:
        return {"loss_fn": hf_loss, "name": "per-token cross-entropy", "confidence": "high",
                "why": "3-D logits with width = num_labels: one decision per position",
                "wrong_if": "sub-word alignment is wrong"}
    if means == "single value":
        return {"loss_fn": regression_loss, "name": "MSE (or Huber)", "confidence": "medium",
                "why": "a single continuous output per example",
                "wrong_if": "it is a binary logit (then BCE), or a reward score (then pairwise)"}
    if means == "hidden state / embedding":
        return {"loss_fn": None, "name": "undetermined: embeddings without pairs", "confidence": "low",
                "why": "the model outputs representations, not predictions",
                "wrong_if": "-", "next": "supply pairs (embedding task) or add a head for your task"}
    return {"loss_fn": None, "name": "CUSTOM ENGINEERING REQUIRED", "confidence": "none",
            "why": f"output means '{means}' with rank {rank}: no standard objective is implied",
            "next": "write loss_fn(model, batch); the rest of the engine is unchanged"}


def derive(model, batch, model_id: str | None = None, task: str | None = None, pairs: bool = False) -> dict:
    """The full first-principles derivation, in the order a human should reason."""
    from . import processors
    from .inspect_model import classify, param_counts
    from .targets import discover
    from .tasks import plan

    cls = classify(model)
    ic, oc = input_contract(model, batch), output_contract(model, batch)
    loss = derive_loss(model, batch, pairs=pairs)
    tgt = discover(model, batch)
    proc = processors.resolve(model_id, model)
    tp = plan(cls["labels"], task) if (task is None or task in TASKS) else None
    return {"1_modality_architecture": cls["labels"], "2_input_contract": ic, "3_output_contract": oc,
            "4_loss": {k: v for k, v in loss.items() if k != "loss_fn"}, "loss_fn": loss["loss_fn"],
            "5_trainable": {"targets": len(tgt["lora_targets"]), "leaf_names": tgt["leaf_names"],
                            "heads": tgt["train_fully_or_freeze"], "do_not_adapt": tgt["do_not_adapt"][:8]},
            "6_processor": processors.describe(proc), "7_task": tp, "processor": proc,
            "params": param_counts(model),
            "missing": ([] if loss["loss_fn"] else ["loss_fn"]) + ([] if proc.encode else ["processor.encode"])}
