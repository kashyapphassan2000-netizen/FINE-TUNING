# 85. Step 2 — Reverse-engineer the forward pass

:::why In one line
Run one real batch through the model with hooks on every layer, and read what actually happens: order, shapes, dtypes, which layers share an input, activation scale, and — after one backward — where the gradients go.
:::

## 85.1 Forward hooks, by hand

```py shapes and order for every Linear / Conv / Embedding
import torch
calls = []
def hook(name):
    def h(mod, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        y = out[0] if isinstance(out, tuple) else out
        calls.append((name, type(mod).__name__, tuple(x.shape), tuple(y.shape), id(x)))
    return h
handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
           if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Embedding))]
with torch.no_grad():
    model(**batch)
for h in handles: h.remove()
for c in calls[:20]: print(c)
```

The last field, `id(x)`, is the key idea of this whole Part: **layers that receive the same tensor object are computed in parallel from one input** — that is how q/k/v and gate/up are recognised without their names.

## 85.2 Backward hooks and gradient inspection

```py after ONE backward pass
loss = model(**batch).loss
loss.backward()
for n, p in model.named_parameters():
    if p.requires_grad:
        g = p.grad
        print(n, None if g is None else float(g.norm()))      # None = this parameter got no gradient
```

`uadapt.trace.gradient_report(model)` does this and flags: trainable tensors with **no** gradient, all-zero gradients, NaN/inf gradients, and frozen tensors that nevertheless received a gradient.

## 85.3 Activation inspection

Per layer, the traced output mean and standard deviation. Watch for: std collapsing to 0 (dead layers), std exploding layer over layer (instability), NaN appearing at one specific layer (the layer where the bug is).

## 85.4 The tool

@file src/uadapt/trace.py

```py
from uadapt.trace import trace_forward
tr = trace_forward(model, batch)       # optionally backward_loss=lambda out: out.loss
print(tr.table(30))                    # order | module | type | in shape | out shape | std
```

## 85.5 The smoke test that must pass before any real training

1. One backward pass: `gradient_report(model)["ok"]` is True.
2. Overfit **one** batch for 30-40 steps: the loss must fall clearly.

If the loss does not fall on a single batch, the objective, the labels or the trainable set is wrong — no amount of data will fix it. Lab 60 runs exactly this on 12 families (chapter 88).

:::warn What the smoke test does NOT prove
A loss falling on one batch proves gradients flow and the objective is wired. It proves nothing about generalisation. That is chapter 103.
:::
