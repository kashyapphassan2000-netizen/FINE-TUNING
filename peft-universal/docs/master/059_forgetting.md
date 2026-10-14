# 59. Catastrophic forgetting

:::why In one line
A model that gets better at your domain can get worse at everything else. If you only measure your domain, you will ship that and not know.
:::

## 59.1 Measured

@file labs/lab17_catastrophic_forgetting.py
@lab lab17_catastrophic_forgetting

A tiny model is first trained on two skills (general + domain), then fine-tuned on the domain alone with full fine-tuning. The domain loss stays at 0.000 while the general loss jumps from 0.004 to 2.987 — the general skill is gone. Adding 25% general "replay" data keeps both at 0.000.

## 59.2 Why it happens, and what reduces it

|Cause|Mitigation|
|---|---|
|the same weights encode both skills, gradients only point at your domain|mix 5-30% general/replay data (strongest, cheapest fix)|
|too many epochs on a narrow dataset|fewer epochs, early stop on a general suite|
|learning rate too high|lower lr|
|too much capacity aimed at one narrow task|lower rank; adapt attention only|
|full fine-tuning|use PEFT: the base weights are frozen, so the damage is bounded|
|adapter always on|keep the adapter separate and turn it off for general traffic (chapter 71)|

:::warn PEFT reduces forgetting, it does not eliminate it
The adapter still changes every layer's output. Measure it. The lab uses full fine-tuning to make the effect visible on a toy; on a real 8B LoRA run the drop is smaller but shows up exactly the same way in a general suite.
:::

## 59.3 The check in your pipeline

```any
python -m peftkit.evaluate --model <base>  --suite domain=... general=... safety=... --out reports/base.json
python -m peftkit.evaluate --model <tuned> --suite domain=... general=... safety=... --out reports/tuned.json
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json
```

The comparison prints `WARNING: domain improved but general/safety regressed -> catastrophic forgetting` when the general delta is below −0.05. Treat that as a release blocker, not a footnote.

:::check
- I have a general suite of at least 50 items that has nothing to do with my domain.
- I ran it before and after.
- If it dropped, I added replay data or reduced epochs/rank, and re-measured.
:::
