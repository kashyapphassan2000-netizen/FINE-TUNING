# 6. Dataset quality: measure it, or you are guessing

:::why The principle
5,000 excellent examples beat 100,000 mediocre ones. The model copies whatever is in the data, including the mistakes, the hedging, and the formatting you did not notice.
:::

## 6.1 Ten axes to measure on every dataset

|Axis|How to measure|Red flag|
|---|---|---|
|Correctness|judge score (ch 17 V1) + human spot check of 30 rows|any wrong answer you find in 30 means ~3%+ wrong overall|
|Diversity|unique 5-gram ratio; count of distinct task verbs; source spread|one template repeated 10,000 times|
|Difficulty|length and reasoning depth distribution|all trivial one-liners|
|Coverage|map rows to your topic list; count per topic|topics with 0 examples: the model will invent|
|Duplicates|`data.quality` exact + near-dup counts|>5% near-duplicates|
|Noise|printable ratio, repetition ratio, truncated answers|repetition loops copied from a teacher model|
|Bias|distribution over entities, dialects, customer types|one product line dominating|
|Contamination|13-gram overlap with test set|any overlap at all is a stop-the-line event|
|Synthetic ratio|count rows with `meta.gen_model`|100% synthetic: read chapter 19 of Volume 1|
|Length|token p50/p95 vs `max_seq_length`|p95 above the limit: answers being cut|

```any One command for the mechanical half
python -m peftkit.data.quality --inputs data/interim/all.jsonl --out data/interim/clean.jsonl \
  --report reports/quality.json --near-dup 0.8 --pii scrub --contamination data/eval/test.jsonl
```

## 6.2 Data leakage, demonstrated

The most common way to fool yourself: eval examples that also appear (or nearly appear) in training.

@file labs/lab18_data_leakage.py
@lab lab18_data_leakage

Same model, same steps, same eval file. The leaked run reports a far better eval loss and token accuracy while knowing nothing extra — it memorised the answer sheet. `data.quality --contamination` catches it, and the doctor (ch 13) fails the run.

:::warn Leakage is rarely a copy-paste
It arrives as: near-duplicates (paraphrases), the same document generating both a training pair and a test question, a public benchmark inside a scraped corpus, or re-using last month's eval set as this month's training data.
:::

## 6.3 The quality gate before every run

:::check
- exact + near duplicates removed; report read.
- test set disjoint from train, verified with `--contamination`.
- 30 rows read by a human, this week, from the current version.
- coverage table has no empty topic that users actually ask about.
- synthetic share known and capped.
:::
