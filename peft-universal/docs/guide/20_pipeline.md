# 20. End-to-end pipeline: building a domain SFT dataset

```text Stages
data/raw/* (10k+ docs any format)            data/seeds.jsonl (20+ human tasks)
   │ ingest.py  (stream, clean, chunk, dedup)       │ self_instruct.py (ROUGE-L filter, stall guard)
   ▼                                                 ▼
chunks.jsonl ── synth.py (grounded QA) ──▶ grounded.jsonl      self_instruct.jsonl ── format.py
   └──────────────────────────────┬─────────────────────────────────────┘
                                  ▼ format.py (normalise → messages, validate)
                                  ▼ quality.py (exact/near dedup, filters, PII, contamination, poisoning)
                                  ▼ judge.py (score ≥ 4)
                                  ▼ format.py (token-length filter with the real tokenizer, split)
                     data/processed/train.jsonl, eval.jsonl   ──▶ (optional) preference.py → DPO pairs
                                  ▼ Argilla review of a sample; human-written eval set kept separate
```

@file scripts/build_dataset.sh

```laptop Run on the laptop (only API calls, no GPU)
$env:DOMAIN = "industrial pump support"
bash scripts/build_dataset.sh        # Git Bash or WSL; or run each python -m line in PowerShell
```

## 20.1 Sizing the dataset

|Goal|Typical size|Notes|
|---|---|---|
|style / format / tone|500-2,000 high-quality rows|quality over quantity (LIMA)|
|domain QA assistant|5k-50k rows|grounded in documents, cover every topic|
|new complex skill|50k+ rows|consider a larger r and more epochs|

These are starting points from common practice, not guarantees. Decide with your eval set: train on 25%, 50%, 100% of the data and plot the eval score.

## 20.2 Quality gates before training

:::check
- `reports/quality.json` reviewed; dropped-row samples read.
- Judge agreement with my labels measured.
- 50 random training rows read by a human.
- Eval set is human-written or human-verified and passed as `--contamination`.
- Synthetic share and teacher model recorded.
- `format.py` run with `--tokenizer <my base model>` so the length filter is exact.
:::
