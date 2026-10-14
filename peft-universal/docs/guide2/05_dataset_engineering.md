# 5. Dataset engineering, end to end

Volume 1 chapters 14, 15 and 20 give the code (`ingest`, `synth`, `self_instruct`, `quality`, `format`, `judge`). This chapter is the engineering discipline around it: the order of operations, and what each stage is protecting you from.

## 5.1 The pipeline, and the reason for every stage

|Stage|Protects against|Tool|
|---|---|---|
|1. Extraction|unreadable formats, silent PDF failures|`data.ingest` (skips and logs bad files)|
|2. Cleaning|control characters, HTML, boilerplate headers/footers|`ingest.clean`|
|3. Normalization|the same fact in five formats confusing the model|your own rules + a fixed system prompt|
|4. Deduplication|memorisation and wasted compute|`data.quality` exact + MinHash near-dup|
|5. PII removal|training a model that leaks customer data|`data.quality --pii scrub`|
|6. Quality filtering|refusal boilerplate, truncated answers, repetition loops|`data.quality`|
|7. Instruction creation|documents are not training examples|`data.synth`, `data.self_instruct`|
|8. Chat formatting|template mismatch at training time|`data.format`|
|9. Token-length analysis|truncation silently deleting answers|`data.format --tokenizer`, `tokenize_inspect`|
|10. Split train/val/test|fooling yourself (ch 6)|`data.format --eval-ratio`, built FIRST|

:::warn Order matters
Deduplicate and filter **before** you split, and build the test set **before** you generate anything. If synthetic data is generated from documents that also produced your test questions, the test set is contaminated at birth.
:::

## 5.2 Three sets, three jobs

|Set|Size|Used for|Rule|
|---|---|---|---|
|train|everything left|the gradient|never look at it for judging quality|
|validation (eval)|100-500 examples|early stopping, hyperparameter choice|you may look at it often; it will get slightly overfitted by your choices|
|test|100-300 examples|the single number you report|human-written or human-verified, opened as rarely as possible|

## 5.3 A practical recipe for a domain assistant

1. Collect the documents. Note the source of each.
2. Write **20-30 real questions by hand** (from support tickets, colleagues, your own use). This becomes the test set, and the seed for Self-Instruct.
3. Generate grounded QA from documents; generate diverse tasks from seeds.
4. Clean, dedup, filter, judge (keep score ≥ 4), and scrub PII.
5. Split, then run the doctor.
6. Train on 25%, 50%, 100% and plot the test score. If the curve is flat, more data is not your problem — quality or method is.

:::check
- Every row can be traced to a source (`meta.source`).
- Test set existed before generation and is excluded via `--contamination`.
- I read 20 random rows and would accept each as a correct answer.
:::
