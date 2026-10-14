# 14. Step 11 — Evaluate, then tune until the gains stop being real

:::why "Reach the global minimum" is not the goal
There is no findable global minimum, and training loss near zero means memorisation. The goal is the best **eval** score on unseen data, reached before the gains fall inside seed noise.
:::

## 14.1 Evaluate first, tune second

```gpu
python -m peftkit.evaluate --model outputs/<run>/adapter --suite domain=... general=... safety=... --out reports/tuned.json
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

Verdict must be PASS: the domain improved and the general/safety suites did not fall. A domain gain that costs general ability is usually a bad trade — mix 5-30% replay data and re-measure.

## 14.2 Hyperparameter search that ends

@file src/peftkit/hpo.py

```gpu
python -m peftkit.hpo --config configs/base.yaml configs/methods/lora.yaml configs/models/my-model.yaml \
    --space '{"train.lr": [5e-5, 1e-4, 2e-4, 5e-4], "method.r": [8, 16, 32], "train.epochs": [1, 2, 3]}' \
    --budget 12 --rungs 20 60 180 --out outputs/hpo --seeds
```

**Random search + successive halving**: every candidate gets a short run, the worst half is killed, survivors get longer runs. In lab 64 six candidates ran at 10 / 25 / 50 steps and most of the budget went to the survivors.

@lab lab64_flow_tools

## 14.3 The stopping rule (this is the answer to "when am I done?")

Run the winner with 2-3 seeds. In the lab: scores 5.003 / 5.026 / 5.082 — a **spread of 0.079**. Any configuration "better" by less than that is not better; it is noise. Stop when:

1. the eval score stops improving by more than the seed spread, **and**
2. the general/safety suites are stable, **and**
3. the failure analysis says the remaining errors are data or task-definition problems, not tuning ones.

## 14.4 The order to tune in (cheapest first)

data quality → learning rate → `max_seq_length` (p95) → effective batch → epochs (watch eval) → rank/alpha → dropout/weight decay → a different method. Never tune two things at once, and record every run.

:::check
- Compare verdict PASS; best checkpoint identified by eval, not by the last step.
- Seed noise measured; the chosen gain is larger than it.
:::
