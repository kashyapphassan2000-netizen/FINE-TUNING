# 64. Preference optimization (DPO) after SFT

:::why In one line
SFT teaches what a good answer looks like. Preference optimization teaches which of two acceptable answers is better — tone, hedging, helpfulness, refusal style.
:::

## 64.1 How DPO works, in plain terms

You have pairs: a prompt, a **chosen** answer, a **rejected** answer. DPO raises the probability of the chosen and lowers the rejected, **relative to a frozen reference copy** of the same model. `beta` controls how far the model may drift from that reference: small beta allows big changes, large beta keeps it close. No reward model and no reinforcement learning loop is required, which is why it replaced RLHF in most small teams.

Reading the metrics:

|Metric|Meaning|Healthy|
|---|---|---|
|`rewards/accuracies`|fraction of pairs where chosen scores above rejected|rises toward 1.0|
|`rewards/margins`|how much better chosen is than rejected|grows steadily|
|`rewards/chosen` / `rewards/rejected`|log-prob shift vs the reference|chosen ≥ 0, rejected going negative|
|loss|always around 0.69 at the start (chance)|falls slowly; alone it says little|

## 64.2 Executed on the tiny model

@file labs/lab23_dpo.py
@lab lab23_dpo

Reward accuracy goes to 1.0 and the margin grows steadily — the model learned the preference. Note the API detail printed at the end: this TRL version's `DPOConfig` has no `max_prompt_length` field, so tutorials that set it will crash.

## 64.3 The pipeline

```text
SFT adapter (this book)  ->  preference pairs (chapter 47: generate K answers, judge, keep clear margins)
                         ->  DPO training on top of the SFT model
                         ->  evaluate BOTH stages separately against the baseline (ch 58)
```

Practical rules:

- **Always SFT first.** DPO on a base model with no instruction ability mostly amplifies noise.
- Use **1k-10k** clean pairs. Pair quality matters more than count; a pair with no real quality gap is noise.
- Keep the SFT eval suites and run them after DPO too: preference training can quietly break format compliance.
- Watch for **length bias**: if your judge preferred longer answers, DPO will make the model verbose. Check mean answer length before and after.
- `beta` 0.1 is the usual start. If the model becomes strange or repetitive, raise beta or lower the learning rate.
- ORPO and KTO are alternatives (ORPO merges SFT and preference in one step; KTO needs only good/bad labels, not pairs). The data pipeline in this book produces the format all of them consume.

:::check
- SFT is done and evaluated before DPO starts.
- Pairs have a real quality gap (margin rule in chapter 47).
- Reward accuracy and margin both rise during training.
- Domain, format and general suites re-run after DPO.
:::
