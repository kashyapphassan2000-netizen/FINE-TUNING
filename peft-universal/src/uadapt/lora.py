"""CUSTOM LoRA for any architecture, no PEFT required.

Supports nn.Linear, nn.Conv1d, nn.Conv2d. Base weights frozen, adapters trainable, adapter-only
state dict save/load, and an EXACT merge:
  Linear : W' = W + s * B @ A
  Conv   : A is a conv(in -> r, kernel k), B is a 1x1 conv(r -> out). Their composition is ONE conv with
           kernel k and weight  W'[o, i, ...] = W[o, i, ...] + s * sum_r B[o, r] * A[r, i, ...]
           (valid for groups=1; grouped/depthwise convs are refused rather than merged wrongly).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)


    # TRANSPARENCY, done correctly (found by lab 60, two bugs deep):
    #  1. model code reads attributes of layers it owns (T5 reads self.wo.weight.dtype) -> must exist;
    #  2. some code never calls forward() and uses .weight FUNCTIONALLY (PyTorch fused fast paths,
    #     nn.MultiheadAttention). Returning the BASE weight there silently drops the adapter.
    # So .weight returns the EFFECTIVE weight W + s*B@A, built live from the parameters, so any
    # functional use is correct AND differentiable.
    @property
    def weight(self):
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)

    @property
    def bias(self):
        return self.base.bias

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.lora_A.T @ self.lora_B.T)

    def merged(self) -> nn.Linear:
        m = self.base
        with torch.no_grad():
            m.weight.add_(self.scale * (self.lora_B @ self.lora_A).to(m.weight.dtype))
        return m


class LoRAConv(nn.Module):
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16):
        super().__init__()
        if base.groups != 1:
            raise ValueError("grouped/depthwise conv: LoRA merge is not exact here - adapt a neighbouring layer instead")
        self.base, self.r, self.scale = base, r, alpha / r
        Conv = type(base)
        self.lora_A = Conv(base.in_channels, r, base.kernel_size, stride=base.stride, padding=base.padding,
                           dilation=base.dilation, bias=False).to(base.weight.device, base.weight.dtype)
        self.lora_B = Conv(r, base.out_channels, 1, bias=False).to(base.weight.device, base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)


    # TRANSPARENCY: see LoRALinear.weight - the effective kernel, so functional use stays correct.
    @property
    def weight(self):
        b = self.lora_B.weight.flatten(1)
        a = self.lora_A.weight.flatten(1)
        return self.base.weight + self.scale * (b @ a).view_as(self.base.weight)

    @property
    def bias(self):
        return self.base.bias

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    def merged(self) -> nn.Module:
        m = self.base
        with torch.no_grad():
            b = self.lora_B.weight.flatten(1)                     # (out, r)
            a = self.lora_A.weight.flatten(1)                     # (r, in * k...)
            delta = (b @ a).view_as(m.weight)
            m.weight += self.scale * delta.to(m.weight.dtype)
        return m


def _set(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def inject(model: nn.Module, targets: list[str], r: int = 8, alpha: float = 16, dropout: float = 0.0,
           freeze_rest: bool = True, train_also: list[str] | None = None) -> dict:
    """Wrap the named modules with LoRA. `targets` are FULL module paths (from targets.discover).
    `train_also`: module paths to train fully (e.g. a new classification head)."""
    if freeze_rest:
        for p in model.parameters():
            p.requires_grad_(False)
    wrapped, skipped = [], []
    for name in targets:
        m = model.get_submodule(name)
        if isinstance(m, nn.Linear):
            _set(model, name, LoRALinear(m, r, alpha, dropout))
            wrapped.append(name)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            try:
                _set(model, name, LoRAConv(m, r, alpha))
                wrapped.append(name)
            except ValueError as e:
                skipped.append((name, str(e)))
        else:
            skipped.append((name, f"unsupported type {type(m).__name__}"))
    for name in train_also or []:
        for p in model.get_submodule(name).parameters():
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {"wrapped": wrapped, "skipped": skipped, "trainable": n_train, "total": n_total,
            "trainable_pct": round(100 * n_train / max(1, n_total), 3)}


def adapter_state_dict(model: nn.Module, also: list[str] | None = None) -> dict:
    keep = {}
    for n, p in model.named_parameters():
        if "lora_" in n or any(n.startswith(a + ".") for a in (also or [])):
            keep[n] = p.detach().cpu().clone()
    return keep


def load_adapter(model: nn.Module, state: dict) -> list[str]:
    missing = model.load_state_dict(state, strict=False)
    return [k for k in state if k in missing.unexpected_keys]


def merge_all(model: nn.Module) -> int:
    """Replace every LoRA wrapper by its merged base layer. Returns how many were merged."""
    count = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, (LoRALinear, LoRAConv)):
            _set(model, name, m.merged())
            count += 1
    return count


def verify_called(model: nn.Module, batch, unwrap_bypassed: bool = False) -> dict:
    """Which LoRA wrappers are executed through forward(), and which are only READ via .weight?

    Wrappers that are not called are not automatically wrong: since .weight returns the effective weight,
    a parent that reads it functionally still uses the adapter. Confirm with trace.gradient_report that
    their lora_A / lora_B receive gradients; if they do not, the layer is not used at all - adapt another."""
    called = set()
    handles = [m.register_forward_hook(lambda mod, i, o, n=n: called.add(n))
               for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv))]
    try:
        with torch.no_grad():
            model(**batch) if isinstance(batch, dict) else model(*batch)
    finally:
        for h in handles:
            h.remove()
    wrappers = [n for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv))]
    not_called = [n for n in wrappers if n not in called]
    if unwrap_bypassed:
        for n in not_called:
            _set(model, n, model.get_submodule(n).base)
    return {"wrappers": len(wrappers), "called": len(called), "not_called": not_called,
            "note": "not_called wrappers are only valid if their parent reads .weight; check their gradients"}
