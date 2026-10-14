"""STEP 2: reverse-engineer the forward pass with hooks. Shapes, dtypes, execution order, shared inputs,
activation statistics and gradient norms - for any nn.Module."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

LEAF_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding, nn.LSTM, nn.GRU, nn.RNN)


@dataclass
class Call:
    order: int
    name: str
    kind: str
    in_shape: tuple | None
    out_shape: tuple | None
    dtype: str | None
    input_id: int | None
    out_mean: float | None = None
    out_std: float | None = None
    has_nan: bool = False
    grad_norm: float | None = None
    params: int = 0


@dataclass
class Trace:
    calls: list = field(default_factory=list)

    def by_name(self) -> dict:
        return {c.name: c for c in self.calls}

    def table(self, limit: int | None = None) -> str:
        rows = [f"{'#':>3} {'module':<44}{'type':<11}{'in':<22}{'out':<22}{'std':>9}"]
        for c in self.calls[:limit]:
            rows.append(f"{c.order:>3} {c.name[-43:]:<44}{c.kind:<11}{str(c.in_shape):<22}{str(c.out_shape):<22}"
                        f"{(c.out_std if c.out_std is not None else float('nan')):>9.4f}")
        return "\n".join(rows)


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    return None


def trace_forward(model: nn.Module, batch: dict | tuple, leaf_types=LEAF_TYPES, backward_loss=None) -> Trace:
    """Run one forward (and optionally backward) pass with hooks on every leaf layer.

    `input_id` is the id() of the tensor the layer received: layers that receive the SAME tensor are
    computed in parallel from one input (q/k/v, gate/up). This is how targets are found without names."""
    tr = Trace()
    handles = []
    counter = [0]
    by_mod = {}

    def fwd_hook(name):
        def h(mod, inp, out):
            x, y = _first_tensor(inp), _first_tensor(out)
            yf = y.detach().float() if y is not None and y.is_floating_point() else None
            c = Call(counter[0], name, type(mod).__name__, tuple(x.shape) if x is not None else None,
                     tuple(y.shape) if y is not None else None, str(y.dtype) if y is not None else None,
                     id(x) if x is not None else None,
                     float(yf.mean()) if yf is not None else None, float(yf.std()) if yf is not None and yf.numel() > 1 else None,
                     bool(torch.isnan(yf).any()) if yf is not None else False,
                     params=sum(p.numel() for p in mod.parameters(recurse=False)))
            counter[0] += 1
            tr.calls.append(c)
            by_mod[mod] = c
        return h

    for name, mod in model.named_modules():
        if isinstance(mod, leaf_types):
            handles.append(mod.register_forward_hook(fwd_hook(name)))
    try:
        out = model(**batch) if isinstance(batch, dict) else model(*batch)
        if backward_loss is not None:
            loss = backward_loss(out)
            loss.backward()
            for mod, c in by_mod.items():
                g = [p.grad for p in mod.parameters(recurse=False) if p.grad is not None]
                c.grad_norm = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else None
    finally:
        for h in handles:
            h.remove()
    return tr


def gradient_report(model: nn.Module) -> dict:
    """After loss.backward(): the silent-failure detector."""
    trainable, with_grad, zero, nan, frozen_with_grad = 0, 0, [], [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable += 1
            if p.grad is None:
                zero.append(n + " (no grad)")
            else:
                with_grad += 1
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    nan.append(n)
                elif float(p.grad.abs().max()) == 0.0:
                    zero.append(n + " (all zeros)")
        elif p.grad is not None:
            frozen_with_grad.append(n)
    lora_zero = [n for n in zero if "lora_A" in n]      # expected at step 0: B is zero-initialised, so
    others = [n for n in zero if "lora_A" not in n]      # dL/dA = 0 until B moves. Not a bug.
    return {"trainable_tensors": trainable, "with_grad": with_grad, "no_or_zero_grad": zero[:20],
            "n_no_or_zero_grad": len(zero), "zero_because_lora_b_is_zero_at_init": len(lora_zero),
            "unexplained_zero_grad": others[:20], "nan_or_inf": nan[:20],
            "frozen_but_has_grad": frozen_with_grad[:20],
            "ok": trainable > 0 and not nan and not frozen_with_grad and with_grad > 0,
            "note": "zero gradients on lora_A at step 0 are normal (B starts at zero); anything in "
                    "unexplained_zero_grad is a real problem: that layer is not on the forward path"}
