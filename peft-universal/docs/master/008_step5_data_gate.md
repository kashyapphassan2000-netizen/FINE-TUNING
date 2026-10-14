# 8. Step 5 — The data gate: when is the dataset actually ready?

:::why One gate, before any GPU
Every hour spent training on bad data is wasted twice: once on the GPU, once debugging a model that was never the problem.
:::

```text PASS ALL OF THESE, OR DO NOT TRAIN
[ ] eval sets built FIRST (domain + general + safety), real, human-checked, frozen, hashed
[ ] contamination check: no eval item (or near-duplicate) inside training
[ ] exact + near duplicates removed; duplicate rate compared with the real baseline
[ ] PII scrubbed or the records dropped; licence and lineage recorded per record
[ ] coverage: every intent x every phrasing x every entity type your users produce
[ ] rare events and refusals present on purpose
[ ] chat template applied with the MODEL'S OWN tokenizer; masking verified on a real row
[ ] token lengths: p95 within max_seq_length; truncation rate < 2%
[ ] 30 rows read by a human this week
[ ] synthetic share known; real data mixed in if the corpus is mostly generated
```

## 8.1 The commands

```any
python -m peftkit.data.quality --inputs data/interim/all.jsonl --out data/interim/clean.jsonl \
    --report reports/quality.json --near-dup 0.8 --pii scrub --contamination data/eval/test.jsonl
python -m peftkit.data.lineage --inputs data/processed/train.jsonl --intended-use commercial
python -m peftkit.tokenize_inspect --tokenizer <model> --file data/processed/train.jsonl --index 0
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/<method>.yaml configs/models/<m>.yaml
```

The doctor is the gate: **zero FAIL**. It catches leakage, truncation, an empty or full loss mask, an untrained end-of-turn token, targets that do not exist in this architecture, a learning rate meant for full fine-tuning, and an impossible AdaLoRA schedule.

:::check
- `peftkit.diagnostics` exits 0.
- `peftkit.data.lineage` returns OK (or REVIEW with a human sign-off).
:::
