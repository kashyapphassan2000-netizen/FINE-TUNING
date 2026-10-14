# 12. Step 9 — Any data, any scale, any domain → train.jsonl

:::why One command, because this is where projects stall
Whatever arrives — a folder of PDFs, a CSV export, 20,000 documents, a Confluence dump — has to become a JSONL file of chat messages with the right template and no leakage.
:::

@file src/peftkit/data/build.py

```any the whole chain in one command
python -m peftkit.data.build --docs data/raw --out data/processed \
   --tokenizer <org/model> --domain "industrial pump support" \
   --mode grounded --eval data/eval/test.jsonl --per-chunk 3 --max-chunks 5000 --max-tokens 4096
```

It runs: **ingest** (txt, md, csv, html, json, jsonl, pdf, docx → cleaned chunks) → **synthesise** grounded Q&A per chunk (or `--mode chunks-only` for continued-pretraining records, no LLM calls) → **quality** (exact + near-duplicate removal, PII scrub, contamination against your eval set) → **format** (chat messages, the model's own template, exact token-length filter, train/eval split). Every stage writes its file, so you can inspect or restart anywhere.

@lab lab64_flow_tools

## 12.1 The target format

```json data/processed/train.jsonl
{"messages": [{"role": "system", "content": "You are a pump support expert."},
              {"role": "user", "content": "What is the maximum pressure?"},
              {"role": "assistant", "content": "The maximum pressure is 6 bar."}]}
```

One line per example. The chat template and the loss mask are applied at training time from **the model's own tokenizer** — never hand-write template tags into the text.

## 12.2 Scale and cost

|Documents|What to do|
|---|---|
|< 1,000|generate 3-5 pairs per chunk; read 30 by hand|
|1,000-20,000|cap with `--max-chunks`; sample chunks across sources so one big document does not dominate|
|> 20,000|ask again whether this is retrieval (facts) rather than fine-tuning (behaviour) — chapter 6.3|

Cost driver: the LLM calls in the synth stage. Cache, cap, and generate 500 first — measure utility — then scale.

## 12.3 Switching the domain

The domain is a **parameter**, not a rewrite: `--domain`, a system prompt, the eval set, and the documents. The same command produces a legal, medical or telecom dataset. What must change with it: the eval set, the refusal examples, the licence/lineage, and the safety suite.

:::check
- `train.jsonl` and `eval.jsonl` exist, and `tokenize_inspect` on row 0 looks right.
- The eval set is not inside the training data (`--eval` was passed).
:::
