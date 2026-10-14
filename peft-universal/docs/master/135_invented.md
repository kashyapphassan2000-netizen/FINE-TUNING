# 135. A modality that does not exist: the final test

:::why The requirement
"Future models that don't exist today." A book cannot prove this with a model someone has already written. So this chapter invents one.
:::

## 135.1 The model

`SeismicFusionNet` (lab 63): a **5×5 grid of seismic sensors** (3 channels per station) fused with a **geology vector**, predicting an **aftershock risk curve over 12 hours**. It has no Hugging Face class, no config, no processor, no tokenizer, no PEFT support and no tutorial. Its objective is a **quantile (pinball) loss** — not shipped by any library in this stack.

```py the only two pieces a human wrote
def quantile_loss(model, batch, qs=(0.1, 0.5, 0.9)):        # 4 lines
    pred = model(batch["grid"], batch["geology"])
    err = batch["risk"] - pred
    q = torch.tensor(qs[len(qs) // 2], device=pred.device)
    return torch.maximum(q * err, (q - 1) * err).mean()

Processor(kind="custom", produces=("grid", "geology"),      # 3 lines
          encode=lambda rec: {"grid": torch.tensor(rec["grid"]),
                              "geology": torch.tensor(rec["geology"])})
```

## 135.2 What the engine did, untouched

@lab lab63_universal_pipeline

From that output's last block:

|Step|Result|
|---|---|
|classification|`custom, vision` (it sees a convolutional grid — reasonable, and it says why)|
|input contract|two positional tensors, observed|
|output contract|rank 2, no loss, "unknown — decide yourself"|
|derived loss|**CUSTOM ENGINEERING REQUIRED** — the honest branch|
|missing|`loss_fn`, `processor.encode` — named before anything ran|
|adapt|LoRA on Linear/Conv, **22.1%** trainable|
|verify|gradients OK, **0** unexplained zero-gradients, loss 0.256 → 0.091, **READY**|
|train|30 steps through the modality-independent loop, final loss 0.066|
|export|5 layers merged, max difference **1.8e-07**|
|serve|CUSTOM ENGINEERING REQUIRED: torch behind FastAPI, then optimise|

## 135.3 What this proves, and what it does not

**Proves**: the engine's derivation, adapter injection, gradient verification, training loop, merge proof and serving advice are genuinely modality-independent — a model invented today, with a loss no library ships, ran the full path without a line of engine code changing.

**Does not prove**: that the model is any good (random weights, synthetic data), or that every future architecture will expose `nn.Linear`/`nn.Conv` layers to adapt. A future model built entirely from packed or fused parameters would fall to the degradation path (chapter 134.3) — selective unfreezing — and a future objective could need a trainer this loop does not have (e.g. an RL rollout loop).

:::check You can claim mastery when
- You can hand the engine a model you invented this morning and get a plan, a smoke test and a merge proof.
- You can name, before training, exactly which pieces it could not derive — and why that is correct.
:::
