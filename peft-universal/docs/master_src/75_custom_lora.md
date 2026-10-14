# 75. Custom LoRA when no framework supports your layer

:::warn CUSTOM ENGINEERING REQUIRED
Used when PEFT does not know the architecture, or you need LoRA on a layer type it does not cover. Everything here is implemented and tested in `uadapt.lora`.
:::

## 75.1 The maths

**Linear.** `y = W x + s·B A x`, with A ∈ R^{r×in}, B ∈ R^{out×r}, B initialised to zero, s = α/r. Merge: `W' = W + s·B A` — exact.

**Convolution.** A = conv(in → r, kernel k, same stride/padding), B = 1×1 conv(r → out), B zero-initialised. A 1×1 conv after a k×k conv is itself a k×k conv, with weight

$$ W'[o, i, :, :] = W[o, i, :, :] + s · Σ_{r} B[o, r] · A[r, i, :, :]

— exact for `groups=1`. Grouped/depthwise convolutions are **refused** rather than merged wrongly (a test asserts this).

@file src/uadapt/lora.py

## 75.2 Two silent failures lab 60 found, and the fix

:::warn Bug 1 — model code reads attributes of the layer it owns
T5's feed-forward block reads `self.wo.weight.dtype`. A wrapper without a `weight` attribute crashes it (`AttributeError: 'LoRALinear' object has no attribute 'weight'`).
:::

:::warn Bug 2 — model code uses `.weight` functionally, without calling the layer
PyTorch's `TransformerEncoderLayer` in eval mode can read `linear1.weight` directly. The first fix for bug 1 returned the **base** weight there — so at inference the adapter was silently ignored, and the "merged" model looked different from the "unmerged" one (difference 3.5). Nothing crashed. Every metric computed in eval mode would have described a model without its adapter.
:::

**The correct design:** a wrapper's `.weight` returns the **effective** weight `W + s·B·A`, computed live from the parameters. Then every access path is right — calling the layer, reading its weight functionally, or checking its dtype — and gradients still flow through the property (a test asserts `F.linear(x, w.weight, w.bias)` equals the wrapper's output and back-propagates into `lora_B`).

**The rule for any custom adapter:** it must be correct both when it is **called** and when its weight is **read**. Test both before trusting it.

## 75.3 The workflow, in code

```py
from uadapt import lora as L
from uadapt.targets import discover
d = discover(model, batch)                                   # chapter 74
info = L.inject(model, d["lora_targets"] + d["conv_candidates"], r=8, alpha=16,
                train_also=d["train_fully_or_freeze"])      # new head trains fully
print(info["trainable_pct"], info["skipped"])
print(L.verify_called(model, batch))                         # which wrappers run via forward()
# ... train (chapter 84) ...
torch.save(L.adapter_state_dict(model), "adapter.pt")        # adapter only: megabytes
with torch.no_grad():
    ref = model(**batch).logits
    L.merge_all(model)                                       # wrappers -> plain layers
    assert torch.allclose(model(**batch).logits, ref, atol=1e-4)   # PROVE the merge
```

## 75.4 What cannot be merged, honestly

|Method|Mergeable?|
|---|---|
|LoRA on Linear, dense Conv|yes, exactly|
|LoRA on grouped/depthwise Conv|not with this construction — adapt a neighbouring layer|
|IA3|yes in principle (per-channel scale folds into the weight); PEFT supports it for supported layers|
|prompt / prefix tuning|no — there is no weight to fold into; ship the virtual tokens with the model|
|bottleneck adapters|no — they add layers; ship them|
|LoRA on packed parameters (LSTM, `nn.MultiheadAttention.in_proj_weight`)|FRAMEWORK LIMITATION with Linear wrapping — re-implement the layer with separate Linears first|
