# 53. Experiment design and tracking

:::why In one line
A run you cannot compare is a run you wasted. One variable, one table, one eval set.
:::

## 53.1 What to record for every run

`run_meta.json` already stores most of it automatically: run id, method, base model, trainable parameters, planned and actual steps, train loss, seconds, peak VRAM, library versions, GPU, full config, and the log history. `experiments.py` adds adapter size on disk and eval scores and puts everything in one table.

|Field|Why it matters later|
|---|---|
|run_id|you will have 40 runs by the end of the week|
|dataset version/hash|the single most common cause of "why can't I reproduce this?"|
|seed|differences smaller than seed noise are not differences|
|method + hyperparameters|the variable you changed|
|eval score(s)|the only reason the run exists|
|VRAM, seconds, adapter MB|the cost side of the decision|
|library versions|behaviour changes between versions (this book found several)|

## 53.2 The grid runner

@file src/peftkit/experiments.py

```any Sweep the rank, score each run on real eval suites
python -m peftkit.experiments --config configs/base.yaml configs/methods/lora.yaml configs/models/<m>.yaml \
  --grid '{"method.r": [8, 16, 32], "train.lr": [1e-4, 2e-4]}' --out outputs/sweep \
  --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl
```

It writes `experiments.csv` and `experiments.md` after every run, so a crash in run 5 does not lose runs 1-4.

## 53.3 Reading results without fooling yourself

- **Seed noise**: run your best config twice with different seeds. If the gap between two configs is smaller than the gap between two seeds, you have learned nothing.
- **One variable**: changing rank and lr together tells you nothing about either.
- **Same eval, every time**: a changed eval set invalidates every earlier row.
- **Cost columns matter**: a +0.5% score for 3x the VRAM is usually a bad trade.
- **Write the conclusion down** in one sentence per sweep ("rank above 16 buys nothing on this data").

:::tip Beyond CSV
`train.report_to: tensorboard` (or `wandb`, `mlflow`) gives live curves. The CSV table is what you show people; the curves are what you debug with.
:::

:::check
- Every run has an id, a dataset version and a seed.
- I ran the same config twice to measure seed noise.
- My conclusions are written as sentences, not remembered.
:::
