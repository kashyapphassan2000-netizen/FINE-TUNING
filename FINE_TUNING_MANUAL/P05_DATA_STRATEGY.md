# P05 — Step 4: Data strategy — what to build, how much, for which domain

**Time:** 45 minutes of thinking that saves days of GPU time. **Cost:** $0.
**GOAL:** before generating a single row: know what the model must learn, build the **evaluation set first**, decide fine-tune vs retrieve, and fix the dataset size and mix.

---

## SECTION 1 — Behaviour or facts? (the decision most projects get wrong)

| The requirement sounds like… | It is really… | Do |
|---|---|---|
| "answer in our format / tone / language" | behaviour | **fine-tune** |
| "extract these 12 fields as JSON" / "classify into our 30 categories" | behaviour (a task) | **fine-tune** |
| "use our medical/legal/engineering terminology correctly" | behaviour + language | fine-tune (CPT first only if T01 §2.4 says so) |
| "know our 20,000 product manuals" / "know the latest prices and policies" | **facts that change** | **retrieve (RAG)**: fine-tuning facts is expensive, goes stale, and hallucinates confidently |
| "answer from our documents, cite them, say when it's not there" | behaviour of **using** retrieved context | RAG + a small fine-tune on (question, retrieved passages) → cited answer / refusal |

```powershell
python labs/lab21_rag_vs_finetune.py
```

What this lab proves (real output): after the manual changed from 6 bar to 9 bar, the fine-tuned model still preferred the old fact (`fine-tuned on the old fact … prefers OLD (stale)`); only a **second training run** fixed it. A trained-in fact is frozen at training time. (Honest limit printed by the lab: its random toy model cannot use retrieved context at all; a real instruct model does, so reproduce the RAG row with an `-Instruct` checkpoint before relying on it.) **Fine-tune for behaviour, retrieve for facts.** Most "make our AI brain switch to our domain" requests are RAG plus a small behaviour fine-tune.

## SECTION 2 — Build the evaluation set FIRST (before any training data)

**WHERE:** VS Code, `FINE-TUNING\peft-universal\data\eval\`
**GOAL:** frozen, human-checked test files that every later run is scored on

Why first: if you write the test after seeing the model's output, you will (unconsciously) write a test it passes.

Step 1. Create the folder `data\eval`.

Step 2. Create `data\eval\test.jsonl`: 100-200 **real** questions from real users (or written by a domain expert), one per line, in this exact shape (the format `peftkit.evaluate` scores):

```json
{"id": "d001", "suite": "domain", "messages": [{"role": "system", "content": "You are a P-200 pump support expert."}, {"role": "user", "content": "What pressure can the P-200 handle continuously?"}], "expected": "6 bar", "must_include": ["6 bar"]}
{"id": "d002", "suite": "domain", "messages": [{"role": "user", "content": "Return the part number for the seal kit as JSON."}], "expected": "{\"part_no\": \"MS-200\"}", "must_match": "^\\{.*\"part_no\".*\\}$"}
```

| Field | Scoring (in `peftkit.evaluate`) |
|---|---|
| `expected` | exact match after normalisation, and "contains gold"; also answer NLL (likelihood of the gold answer) |
| `must_include` | every listed phrase must appear |
| `must_match` | regex the answer must match (formats: JSON, dates, codes) |
| `suite` | informational only: the report's suite name comes from the command line (`--suite domain=data/eval/test.jsonl general=...`). The verdict treats suites named `general` and `safety` as the forgetting check |
| last `user` message | the prompt that is sent; a leading `system` message is sent as the system prompt |

Step 3. Create `data\eval\general.jsonl` (30-100 general questions the base model already answers well: arithmetic, common knowledge, instruction following) and `data\eval\safety.jsonl` (30-50 prompts the model must refuse or handle carefully in your domain). These two detect **catastrophic forgetting** (T04 §5).

Step 4. **Hold out entities:** if your domain has products/patients/customers/machines, put some of them **only** in the test set. Testing on unseen entities is the real test.

Step 5. Freeze it: never edit these files after training starts. If you must change them, bump a version (`test_v2.jsonl`) and re-score the baseline. Back them up **privately**: your `FINE-TUNING` repo is **public**, so `data/eval/` is in `.gitignore` on purpose. Real user questions must not be published. Keep the eval set in a **private** Hugging Face dataset repo (P14 Section 4 with `--private`), or private storage:

```powershell
python -m peftkit.publish --src data/eval --hub HF_USER/SLUG-eval --repo-type dataset --private
```

> **WATCH OUT:** never generate the test set with the same LLM and documents you use to generate training data. A high score then only proves the model copies the generator.

## SECTION 3 — How much data, and what mix

| Goal | Rows to start with | Notes |
|---|---|---|
| format / style / tone only | 200-1,000 | diversity of inputs matters more than count |
| a narrow task (extraction, classification) | 1,000-5,000 | cover every label/field; balance rare classes |
| a domain assistant (Q&A, troubleshooting) | 3,000-20,000 | coverage: intents × entities × phrasings |
| a broad multi-task assistant | 20,000-200,000 | FLAN-style task diversity (T04 §2) |
| continued pre-training (CPT) | 100M+ **tokens** of raw text | T01 §2.4 |

**The mix (percentages of the training set):**

| Slice | % | Why |
|---|---|---|
| core domain tasks | 60-80% | the job |
| hard cases: ambiguous, multi-step, long inputs | 5-15% | where users actually fail |
| refusals / "not in the documents" / clarifying questions | 5-10% | stops confident hallucination |
| replay: general instruction data (or the previous version's data) | 5-30% | prevents forgetting (lab 17, lab 54) |

**Stopping rule for adding data:** train at 25%, 50%, 100% of your data and score each on the test set (P12 learning curve). If 50% → 100% gains less than the seed noise (P12 §4), more of the **same** data will not help; add **different** data (new intents, entities, phrasings) instead.

## SECTION 4 — Changing the domain of a model (the recipe)

The domain is a **parameter** of the pipeline, not a rewrite. To retarget from "pump support" to "clinic operations" or "insurance claims" you change exactly these:

| Change | Where |
|---|---|
| documents or real sample | `data/raw/` (P06) or your real export (P07) |
| `--domain` and the system prompt | `peftkit.data.build`, `peftkit.data.records --system` |
| the **eval sets** (domain, general, safety) | `data/eval/` (this Part, Section 2) |
| refusal examples and safety suite | domain-specific: medical advice limits, legal disclaimers, financial advice rules |
| licence / lineage / PII policy | `peftkit.data.lineage`, `--pii scrub|drop` |
| the run config | `configs/models/<SLUG>.yaml` (P10) |

Domain-specific rules for healthcare, legal, finance and others are in P16.

## SECTION 5 — How to make the model learn continuously (iterative fine-tuning)

A model does not "keep learning" on its own after deployment. Continuous learning is a **loop you run**, version by version:

```text
v1: train on dataset v1 -> evaluate -> publish HF_USER/SLUG revision v1
 -> collect: real user questions, thumbs-up/down, failures found by evaluation, new documents
 -> curate: dedup against v1, fix wrong answers, add the failures as NEW training rows (and new TEST rows)
 -> dataset v2 = new rows + REPLAY (20-50% of v1, so v1 skills are not forgotten)
 -> fine-tune again (from the base + all data, or from v1's merged model with lower lr)
 -> evaluate v2 on the v1 test set AND the new test rows -> publish only if it beats v1 everywhere that matters
```

| Rule | Why |
|---|---|
| version everything: `DATASET_NAME-v1.0`, model revision `v1`, config, seed | you must be able to roll back |
| keep the old test set; add new tests, never delete old ones | otherwise you cannot see regressions |
| always replay old data | catastrophic forgetting (lab 17) |
| thumbs-up/down data → KTO; side-by-side preferences → DPO | T05 §6, P13 |
| facts that change weekly → update the RAG index, not the model | retraining for facts is the expensive, stale way |

---

## CHECKPOINT

- [ ] I wrote down which parts are behaviour (fine-tune) and which are facts (retrieve).
- [ ] `data/eval/test.jsonl` (100-200 real questions), `general.jsonl` and `safety.jsonl` exist and are backed up privately.
- [ ] Some entities exist only in the test set.
- [ ] I chose a starting dataset size and the mix percentages.
