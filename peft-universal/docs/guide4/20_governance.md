# 20. Governance, cost and the things that bite in month six

## 20.1 Versioning and reproducibility

Version four things together: the **contract**, the **real seed snapshot**, the **generator config + seed**, and the **gold test set**. A dataset version is the tuple. Store the report alongside. If you cannot rebuild version 3 byte-for-byte, you cannot explain a regression between versions 3 and 4.

## 20.2 Provenance on every record

`source`, `generator_model`, `generator_version`, `prompt_version`, `contract_version`, `parent_batch`, `judge_scores`, `created_at`. This is what lets you delete exactly the poisoned batch instead of the whole corpus, prove to an auditor where a record came from, and measure your synthetic share over time (chapter 16).

## 20.3 Poisoning: accidental and deliberate

|Direction|Looks like|Defence|
|---|---|---|
|accidental|the teacher model invented facts, repeated itself, drifted in style|judges + fidelity gates + human sample audit|
|deliberate|crafted records that flip a behaviour behind a trigger|provenance, batch-level review, blocklist scans (Volume 1, ch 19), no unreviewed external contributions|

Both are handled by the same discipline: nothing enters the corpus without a batch identity and a gate result.

## 20.4 Cost control

|Lever|Effect|
|---|---|
|cache every LLM call by prompt hash|the single biggest saver on repeated runs|
|generate structured fields with the copula, text only with the LLM|10-100× fewer tokens than generating whole records|
|batch records per call|fewer, larger calls|
|route by difficulty: cheap model for bulk, strong model for judging and hard cases|cost falls, quality holds|
|filter early: schema and dedup before judging|do not pay a judge to score malformed records|
|generate 500, measure utility, then scale|the most expensive mistake is generating a million rows nobody evaluated|
|self-host for bulk generation once volume is steady|fixed cost replaces per-token cost|

Track tokens per accepted record. That is the real unit cost, and it is always worse than tokens per generated record, because of the rejects.

## 20.5 CI for datasets

Treat a dataset like code: a pull request runs contract validation, fidelity, privacy and gates on a sample, fails the build on any hard gate, and publishes the report. Weekly scheduled job: re-run utility against a fresh real slice and the drift monitor. This is the difference between a synthetic data system and a pile of JSONL files.

:::check
- Every dataset version = contract + seed snapshot + generator config + gold set.
- Provenance on every record.
- Gates run in CI, not by hand.
- Cost tracked per accepted record.
:::
