# 46. Instruction dataset generation: grounded QA, Self-Instruct, Alpaca

## 46.1 Three generators, one output format

|Generator|Input|Best for|Module|
|---|---|---|---|
|Grounded QA|your document chunks|domain knowledge, company data|`synth.py`|
|Self-Instruct|10-200 human seed tasks|task diversity, skills|`self_instruct.py`|
|Alpaca-style answers|instructions|turning instructions into (input, output) pairs|`self_instruct.answer_all`|

All write jsonl that `format.py` turns into messages.

## 46.2 Ingest: any folder of documents → chunks

@file src/peftkit/data/ingest.py

It streams file by file, so 10 or 100,000 documents use the same memory; a corrupt file is logged and skipped (tested with a fake PDF). For scanned PDFs, OCR them first (pypdf only extracts embedded text).

```any
python -m peftkit.data.ingest --src data/raw --out data/interim/chunks.jsonl --chunk-chars 2000 --overlap 200
```

## 46.3 Grounded QA from chunks

@file src/peftkit/data/synth.py

The prompt tells the teacher to write question-answer pairs answerable **only** from the passage. Each row keeps `meta.chunk_id` and `meta.source`, which makes the output resumable (re-running skips finished chunks, tested) and auditable (you can trace every example to a document).

## 46.4 Self-Instruct (Wang et al., 2022, arXiv 2212.10560)

The paper started from 175 human-written seed tasks and bootstrapped with GPT-3: each round samples 8 in-context tasks (6 human, 2 machine-generated), asks for new instructions, and keeps a new one only if its ROUGE-L similarity with every existing instruction is below 0.7; it also drops tasks a text model cannot do (images, graphs). Then it generates inputs and outputs and filters again.

$$ ROUGE-L F1 = 2 · P · R / (P + R),   P = LCS / |x|,   R = LCS / |y|

LCS = longest common subsequence of words. `self_instruct.py` implements exactly this loop, with a domain description added to the prompt, and one production addition: **a stall guard**. When the teacher keeps proposing near-duplicates, no new instructions pass the filter; without a guard the loop keeps paying for calls up to `max_rounds`. Found by a test, now stopped after `max_stall` empty rounds.

@file src/peftkit/data/self_instruct.py

## 46.5 Alpaca: scaling Self-Instruct

Stanford Alpaca (Taori et al., 2023) used a simplified Self-Instruct pipeline with text-davinci-003 to produce 52K instruction-following examples and fine-tuned LLaMA-7B on them. Simplifications described by the authors include generating a batch of instructions per call and dropping the classification/non-classification split. Its output format (`instruction`, `input`, `output`) became a de facto standard; `format.py` reads it directly. Lessons that still apply: diversity comes from the seed pool and filters, and teacher mistakes are copied into the student.

```any Seeds file: data/seeds.jsonl (write 20+ real tasks from your domain)
{"instruction": "Explain how to check an impeller for cavitation damage."}
{"instruction": "Draft a reply to a customer whose warranty claim lacks a serial number."}
```

```any Run
python -m peftkit.data.self_instruct --seeds data/seeds.jsonl --domain "industrial pump support" --n 300
```

```out Shape of the report (fields are real; numbers depend on your teacher)
{"proposed": ..., "length": ..., "banned_word": ..., "bad_start": ..., "too_similar": ..., "rounds": ..., "stalled": false, "written": ..., "usage": {"calls": ..., "prompt_tokens": ..., "completion_tokens": ..., "failures": 0}}
```

## 46.6 The shared API client

@file src/peftkit/data/llm.py

`OPENAI_BASE_URL` is read from `.env`, never hard-coded, so Euron, vLLM (your own model), or any OpenAI-compatible provider works. Retries on 429 and 5xx with backoff; one re-ask on malformed JSON; token usage is counted so you can price each stage.

:::check
- `data/seeds.jsonl` has at least 20 human tasks from my domain.
- I ran 10 examples with `--limit 10` / `--n 10` and read them before scaling up.
:::
