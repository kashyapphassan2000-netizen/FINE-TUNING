# 127. Data provenance, lineage and licence controls

:::why In one line
In an enterprise, "where did this training example come from and what may we do with it?" must be answerable per record, not per dataset.
:::

@file src/peftkit/data/lineage.py

## 127.1 The fields on every record

```json required
{"source_id": "manual-batch-3", "source_type": "human|scraped|licensed|synthetic|customer",
 "license": "apache-2.0", "dataset_version": "1.4", "created_at": 1758000000, "record_hash": "a1b2..."}
```

```json optional but valuable
{"generator_model": "...", "generator_version": "...", "prompt_version": "p7", "transformation": "paraphrase",
 "quality_score": 4.2, "filter_reason": null, "parent_batch": "b12", "consent": "recorded-2026-05-11",
 "retention_until": 1790000000}
```

```any
python -m peftkit.data.lineage --inputs data/processed/train.jsonl --intended-use commercial
```

## 127.2 What the checks catch

- **Missing lineage**: any record without the required fields blocks training (a test plants this).
- **Licence conflicts**: CC-BY-NC data with commercial intent → **BLOCK** (the CC-BY-NC trap: Alpaca-derived data is not usable commercially).
- **Share-alike** licences (CC-BY-SA, ODbL, GPL) → **REVIEW**: they may impose terms on what you distribute.
- **Synthetic data**: the **generator's** terms also apply — several vendors restrict training competing models.
- **Customer data**: contract, consent, retention and deletion obligations.
- Unknown licences are treated as non-commercial until identified.

The tool returns `OK` / `REVIEW` / `BLOCK` with the reasons, and says plainly: engineering control, not legal advice — a human signs it off.

## 127.3 Why lineage pays for itself

|Situation|With lineage|Without|
|---|---|---|
|a batch is found to be poisoned or wrong|delete exactly that `parent_batch` and retrain|re-inspect everything|
|a customer invokes deletion|find and remove their records|you cannot prove you did|
|a licence turns out to be incompatible|remove that `source_id`, quantify the loss|the whole corpus is suspect|
|a regression appears between versions|diff dataset versions|guesswork|
|an auditor asks "what is this model trained on?"|`summarise()`: licences, source types, synthetic share|no answer|
