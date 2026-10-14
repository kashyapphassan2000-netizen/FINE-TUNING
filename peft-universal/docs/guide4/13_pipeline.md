# 13. The whole thing in one command

@file src/synthkit/pipeline.py

```any
python -m synthkit.pipeline --real data/real.jsonl --contract contracts/support_tickets.yaml \
  --numeric severity first_response_minutes --categorical customer_tier issue_type \
  --label churned --n 5000 --smoothing 0.25 --out reports/run7
```

It splits your real data 60/20/20 (train / privacy holdout / gold), profiles the train split only, fits the generator, enforces contract types and bounds, then runs every check in chapters 8-12 and writes `synthetic.jsonl` plus a full `report.json`.

## 13.1 The full run, start to finish

@file labs/lab40_end_to_end.py

## 13.2 What to keep from every run

|Artifact|Why|
|---|---|
|`synthetic.jsonl`|the data|
|`report.json`|fidelity, privacy, utility, gates — your evidence|
|contract + version|what the data promised to be|
|profile.json of the real train split|what it was mimicking|
|generator config + seed|reproducibility|
|the gold test set hash|proof it was not contaminated|

With those six, anyone can reproduce or audit the dataset. Without them, you have a file of numbers nobody can defend.
