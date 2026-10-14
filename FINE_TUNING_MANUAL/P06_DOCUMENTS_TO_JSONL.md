# P06 — Step 5: Your PDFs, Excel, Word, CSV, slides → train.jsonl

**Time:** 1 hour (plus LLM API time for big corpora). **Cost:** $0 in `chunks-only` mode; LLM API tokens in `grounded` mode (cents per 100 chunks with a small model; you set the cap).
**GOAL:** any pile of documents becomes a clean, leak-free, chat-format `train.jsonl` + `eval.jsonl`, with every stage's file kept so you can inspect and restart.

## What the one command does

```text
data/raw/  (any mix of files, any sub-folders)
  1/4 ingest   read every file -> clean text -> chunks on paragraph/sentence boundaries (with overlap) -> exact-dup removal
  2/4 synth    grounded: an LLM writes N question/answer pairs per chunk, answered ONLY from that chunk
               chunks-only: no LLM; each chunk becomes a continued-pretraining style record
  3/4 quality  exact + near-duplicate removal, PII scrub, refusal/boilerplate filter, contamination vs your eval set
  4/4 format   chat messages, the MODEL'S OWN chat template, exact token-length filter, train/eval split
data/processed/train.jsonl, eval.jsonl        (every stage's file kept in data/interim/<out folder name>/)
```

| Format | Read as | Notes |
|---|---|---|
| `.pdf` | text layer of each page | **scanned PDFs have no text layer** → run OCR first: `ocrmypdf in.pdf out.pdf` |
| `.docx` | paragraphs | old `.doc`: open in Word → Save As `.docx` |
| `.xlsx` / `.xlsm` | every row → `[sheet] column: value \| column: value`, rows grouped into chunks, never split mid-row | old `.xls`: Save As `.xlsx`. Formulas are read as their computed values |
| `.csv` / `.tsv` | same as Excel | a `text` column, if present, is used as-is |
| `.pptx` | text of every slide, `Slide N:` prefix | speaker notes are not read |
| `.html` / `.htm` | text with scripts/styles removed | |
| `.txt` / `.md` | as-is | |
| `.json` / `.jsonl` | the `text` or `content` field, or the whole object | |

---

## SECTION 1 — Put your documents in place

**WHERE:** File Explorer
**GOAL:** all source files under `FINE-TUNING\peft-universal\data\raw\`

Step 1. Copy your documents into `data\raw\` (sub-folders are fine; the source path is recorded per chunk).

Step 2. (To practise first with the sample set: a PDF manual, an Excel parts list, a Word FAQ.) Paste and press Enter:

```powershell
python examples/sample_docs/make_sample_docs.py
```

**YOU SHOULD SEE:**

```text
wrote 3 documents to ...\examples\sample_docs\raw
wrote ...\examples\sample_docs\faq.xlsx
```

> **WATCH OUT:** your repo is public, and `data/raw/` is in `.gitignore` so your documents are never committed. Keep it that way: never `git add -f` real documents.

## SECTION 2 — Dry run without an LLM (chunks-only)

**WHERE:** PowerShell, `(.venv)` active
**GOAL:** see that every file is read and chunked correctly, at $0

Step 1. Paste and press Enter (use `--docs data/raw` and your `MODEL_ID` for real data):

```powershell
python -m peftkit.data.build --docs examples/sample_docs/raw --out outputs/docs_chunks --tokenizer outputs/tiny --domain "industrial pump support" --mode chunks-only --chunk-tokens 120
```

**YOU SHOULD SEE (real output):**

```text
1/4 ingest
    {'files': 3, 'skipped_files': 0, 'chunks': 7, 'dupes': 0} -> outputs/interim/docs_chunks/chunks.jsonl
2/4 chunks-only mode: continued-pretraining style records, no LLM calls
3/4 quality: dedup, near-dup, PII, contamination
    {'read': 7, 'kept': 7} -> outputs/interim/docs_chunks/clean.jsonl
4/4 format: chat template check, token-length filter, split
    {'read': 7, 'bad_shape': 0, 'invalid': {}, 'dupes': 0, 'too_long': 0, 'train': 7, 'eval': 0}
ready: outputs/docs_chunks/train.jsonl (7 rows) and outputs/docs_chunks/eval.jsonl (0 rows)
```

Step 2. Look at the chunks (each one shows its source file and where it starts):

```powershell
python -c "import json; [print(json.loads(l)['source'], '|', json.loads(l)['text'][:90].replace(chr(10),' / ')) for l in open('outputs/interim/docs_chunks/chunks.jsonl', encoding='utf-8')]"
```

**YOU SHOULD SEE (real output):**

```text
P-200_manual.pdf#0.0 | Model P-200 centrifugal pump - Operating manual, section 4: pressure limits. / The maximum c
P-200_manual.pdf#0.1 | Prime the casing before the first start. Never run the pump dry for more than 30 seconds. /
spare_parts.xlsx#0.0 | [spare_parts] part_no: MS-200 | description: mechanical seal kit | fits_model: P-200 | rep
spare_parts.xlsx#0.1 | [spare_parts] part_no: RV-7 | description: relief valve 7 bar | fits_model: P-200 | replac
support_faq.docx#0.0 | P-200 support FAQ / Why does the relief valve open? / Discharge pressure exceeded 7 bar; check
```

| If you see… | It means… | Do this |
|---|---|---|
| `skipped_files: N` and `[skip] file: error` lines | a corrupt or password-protected file | fix or remove it; one bad file never stops the run |
| a PDF with 0 chunks | scanned images, no text layer | `ocrmypdf` it, re-run |
| `chunks` far fewer than expected | files under `--min-chars` (200) or exact duplicates | normal for tiny files; check `dupes` |
| chunks that are half a table | `--chunk-tokens` too small for your rows | raise it (700 is the default) |

**Choosing `--chunk-tokens`:** the default 700 (≈2,800 characters) gives the LLM enough context to write good questions. Use 300-500 for FAQ-like documents and 1,000+ for long technical sections. Overlap is 200 characters and always starts at a clean boundary.

## SECTION 3 — The real run: grounded Q&A from your documents

**WHERE:** PowerShell; `.env` has `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `SYNTH_MODEL` (P01 Section 6)
**GOAL:** `data/processed/train.jsonl` with questions a real user would ask, answered only from your documents

Step 1. Cap the cost first: `--max-chunks 50` for a trial run, then remove the cap. Paste and press Enter:

```powershell
python -m peftkit.data.build --docs data/raw --out data/processed --tokenizer MODEL_ID --domain "DOMAIN" --mode grounded --per-chunk 3 --max-chunks 50 --max-tokens 4096 --eval data/eval/test.jsonl --system "You are a helpful DOMAIN assistant."
```

| Flag | Value | Why |
|---|---|---|
| `--tokenizer` | your `MODEL_ID` | exact token lengths with the model's own template |
| `--domain` | e.g. `"industrial pump support"` | goes into the generation instruction |
| `--per-chunk` | 3 (1-5) | Q&A pairs per chunk; more = more phrasings, more cost |
| `--max-chunks` | 50 for the trial, 0 = all | caps LLM calls (cost) |
| `--eval` | your `data/eval/test.jsonl` | **contamination guard**: any training row sharing a 13-gram with a test item is dropped |
| `--system` | the system prompt you will SERVE with | written into every row (the generator's own instruction is never written into the rows) |
| `--max-tokens` | ≤ the context you will train with | rows longer than this are dropped, never truncated |

**YOU SHOULD SEE (real output of the same command against the toolkit's fake OpenAI-compatible test server; with a real LLM the questions and answers are about your documents):**

```text
1/4 ingest
    {'files': 3, 'skipped_files': 0, 'chunks': 7, 'dupes': 0} -> outputs/interim/docs_grounded/chunks.jsonl
2/4 synthesise grounded Q&A (LLM calls)
    -> outputs/interim/docs_grounded/pairs_grounded.jsonl
3/4 quality: dedup, near-dup, PII, contamination
    {'read': 14, 'kept': 14} -> outputs/interim/docs_grounded/clean.jsonl
4/4 format: chat template check, token-length filter, split
    {'read': 14, 'bad_shape': 0, 'invalid': {}, 'dupes': 0, 'too_long': 0, 'train': 14, 'eval': 0}
ready: outputs/docs_grounded/train.jsonl (14 rows) and outputs/docs_grounded/eval.jsonl (0 rows)
```

Step 2. If it stops halfway (network, rate limit), **run the same command again**: chunks already generated are skipped (resumable).

| If you see… | It means… | Do this |
|---|---|---|
| `ERROR: 0 training rows survived` + `drop_repetitive` | near-identical boilerplate documents | add varied documents or raise `--chunk-tokens` |
| `failed_chunks` high | API errors, wrong model name, rate limits | check `.env`; the client retries 429/5xx with backoff 4 times |
| `drop_benchmark_contamination: N` | N rows overlapped your test set | good: the guard worked. If N is large, your test set is too close to the documents |
| `too_long` large | rows exceed `--max-tokens` | raise it (≤ your training context) or use smaller chunks |
| `KeyError` / JSON errors from the provider | the model does not follow the JSON instruction | use a stronger `SYNTH_MODEL` |

## SECTION 4 — Inspect, then read 30 rows (do not skip this)

**WHERE:** PowerShell and VS Code
**GOAL:** proof that the data is correct and the model will see what you think

Step 1. See one example exactly as the model sees it, token by token, with the loss mask:

```powershell
python -m peftkit.tokenize_inspect --tokenizer MODEL_ID --file data/processed/train.jsonl --index 0 --limit 60
```

**YOU SHOULD SEE (last line):** `total N tokens, trained on M (x%), EOS trained: True`. Fix anything else before going on (T02 §3).

Step 2. Read the quality report: `data/interim/processed/quality.json` counts every dropped row by reason.

Step 3. Open `data/processed/train.jsonl` in VS Code and read **30 random lines**. For each, ask: is the answer correct, complete, and something a real user would want? If more than 2 of 30 are wrong, fix the generation (a stronger model, a better `--domain`, smaller chunks) before training. Training on wrong answers teaches wrong answers.

## SECTION 5 — Combine sources, add an LLM-judge filter, record lineage

**WHERE:** PowerShell
**GOAL:** one final dataset from several sources, filtered by quality, every row traceable

Step 1. (Optional) Keep only rows an LLM judge scores ≥ 4/5. Use a **different** model than the generator if you can:

```powershell
python -m peftkit.data.judge --input data/processed/train.jsonl --out data/interim/judged.jsonl --min-score 4 --judge-model JUDGE_MODEL_NAME
```

Step 2. Merge several sources (documents + your own human-written rows + replay data) into one validated, deduplicated split:

```powershell
python -m peftkit.data.format --inputs data/interim/judged.jsonl data/human/my_examples.jsonl data/replay/general.jsonl --out-dir data/processed --eval-ratio 0.05 --tokenizer MODEL_ID --max-tokens 4096
```

Step 3. Stamp lineage onto every row (rows from `peftkit.data.build` have none yet; rows from `peftkit.data.records` already do). `--license owned` = your own/your employer's documents; use the real licence for third-party sources (`cc-by-4.0`, `apache-2.0`, …):

```powershell
python -m peftkit.data.lineage --inputs data/processed/train.jsonl --stamp --source-id DATASET_NAME-docs --source-type synthetic --license owned --dataset-version 1.0 --generator-model SYNTH_MODEL_NAME
```

**YOU SHOULD SEE:** `stamped N rows -> data/processed/train.jsonl`

Step 4. Check licences and lineage:

```powershell
python -m peftkit.data.lineage --inputs data/processed/train.jsonl --intended-use commercial
```

**YOU SHOULD SEE (real output on the stamped sample):** `"ok": true` and `"decision": "REVIEW"` with the warning *"synthetic records present: the GENERATOR's terms of service also apply"*. A human reads it and signs off. `BLOCK` = a source's licence forbids your use (e.g. CC-BY-NC data for a commercial model).

## SECTION 6 — Your spreadsheet ALREADY has questions and answers

**WHERE:** PowerShell
**GOAL:** an Excel/CSV FAQ, ticket export or Q&A sheet → chat JSONL in one command, no LLM

Step 1. Paste and press Enter (column names in `{braces}` must match your header row; repeat `--user` for several phrasings):

```powershell
python -m peftkit.data.records --input examples/sample_docs/faq.xlsx --name pump-faq --task template --user "{question}" --assistant "{answer}" --system "You are a P-200 pump support expert." --source-type human --out outputs/datasets
```

**YOU SHOULD SEE (real output):**

```text
{ "name": "pump-faq", "version": "1.0", "task": "template", "rows": 3, "conversations": 3, "train": 3, "eval": 0,
  "dir": "outputs/datasets/pump-faq-v1.0" }
{"messages": [{"role": "system", "content": "You are a P-200 pump support expert."}, {"role": "user", "content": "What is the max pressure of the P-200?"}, {"role": "assistant", "content": "6 bar continuous; peaks up to 7 bar for 10 seconds."}], "meta": {"source_id": "pump-faq:faq.xlsx", "source_type": "human", ...}}
```

For a ticket export with columns `subject`, `body`, `resolution`: `--user "{subject}\n\n{body}" --assistant "{resolution}"`. For several sheets: `--sheet SheetName`.

---

## CHECKPOINT

- [ ] Every file in `data/raw` was read (`skipped_files` explained), and chunks look right.
- [ ] `data/processed/train.jsonl` exists; `tokenize_inspect` shows a sane trained % and `EOS trained: True`.
- [ ] The contamination guard ran against `data/eval/test.jsonl`.
- [ ] I read 30 random rows and ≤ 2 were wrong.
- [ ] Lineage check is OK or REVIEW-signed-off.
