# 9. Evaluation: the biggest missing piece

:::why The rule
"The loss went down" is not a result. The only question that matters is: **does the model do better on examples it has never seen, measured the same way before and after?**
:::

## 9.1 The shape of an honest evaluation

```text
base model ──run the SAME suites──> baseline scores
                                         │
fine-tuned model ──same suites──> after scores
                                         ▼
                            deltas + named regressions + verdict
```

You must evaluate the **base model first**. Without a baseline you cannot tell improvement from noise, and you cannot tell whether fine-tuning was needed at all.

## 9.2 Four suites, not one

|Suite|Contains|Catches|
|---|---|---|
|domain|the task you trained for|did it work|
|general|ordinary questions outside the domain|catastrophic forgetting (ch 10)|
|format/regression|items with a strict output shape, past bugs|"it answers well but broke the JSON"|
|safety|refusal-worthy prompts, prompt injection|the fine-tune eroding refusals|

Eval row format (`must_include` / `must_match` make automatic scoring possible):

```json data/eval/test.jsonl
{"id": "p1", "suite": "domain", "messages": [{"role": "user", "content": "What is the maximum pressure?"}],
 "expected": "6 bar", "must_include": ["6 bar"]}
{"id": "f1", "suite": "format", "messages": [{"role": "user", "content": "Return the limits as JSON."}],
 "must_match": "^\\s*\\{.*\\}\\s*$"}
```

## 9.3 The harness

@file src/peftkit/evaluate.py

```any Baseline, then tuned, then compare
python -m peftkit.evaluate --model meta-llama/Llama-3.1-8B-Instruct \
   --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/base.json
python -m peftkit.evaluate --model outputs/<run>/adapter \
   --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/tuned.json
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

It reports per-suite score and mean answer NLL, the **list of items that got worse** (regressions, by id, with the new prediction), and a verdict: `PASS`, `FAIL: the domain did not improve`, or `WARNING: domain improved but general/safety regressed -> catastrophic forgetting`. The CLI exits non-zero unless the verdict is PASS, so it can gate a deployment.

## 9.4 Which metric for which task

|Task|Metric|
|---|---|
|closed QA with a short answer|exact match / contains gold|
|structured output|schema or regex validity rate|
|classification|accuracy, per-class recall|
|style, tone, long answers|LLM-as-judge pairwise vs the baseline (Volume 1, ch 17), with humans on a sample|
|anything ambiguous|answer NLL as a stable secondary signal|
|latency/cost|tokens/sec and p95 latency under real load|

:::warn Judge discipline
Use a different model as judge than the one you trained from, swap A/B positions, and calibrate the judge against 50 human labels before trusting it. A judge that prefers longer answers will happily certify a worse model.
:::

## 9.5 Human evaluation, cheaply

50 items, two anonymised answers side by side (baseline vs tuned), one question: "which would you send to a customer?" Count wins. This takes an hour and settles arguments that a week of metric discussion will not.

:::check
- Baseline measured BEFORE training, on the same suites.
- General/safety suites included, not just the domain.
- Regression list read item by item.
- One sentence written: "X went from A to B on N unseen examples."
:::
