# T02 — Data Preparation for Fine-Tuning

**WHERE:** read anywhere; labs run on your PC from `FINE-TUNING\peft-universal` with `(.venv)` active.
**GOAL:** understand exactly what one training example is, how the model sees it token by token, which tokens are trained, and how a dataset is cleaned. These are the five ideas that decide 80% of fine-tuning quality.

---

## 1. Dataset formats (instruction pairs, chat format)

Every format below says the same thing: "given this, answer that". The toolkit auto-detects all of them (`peftkit.data.format`) and converts them to the one format trainers want: **chat `messages`**.

| Format | One line of JSONL | Where it comes from |
|---|---|---|
| **messages** (target format) | `{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}` | OpenAI/TRL standard; what you train on |
| ShareGPT | `{"conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}` | older open chat datasets |
| Alpaca (instruction pairs) | `{"instruction":"...","input":"...","output":"..."}` | Stanford Alpaca, Self-Instruct |
| prompt / completion | `{"prompt":"...","completion":"..."}` | older OpenAI fine-tuning, completion-only tasks |
| question / answer | `{"question":"...","answer":"..."}` | QA exports, FAQ spreadsheets |
| preference (DPO/ORPO) | `{"prompt":[...],"chosen":[{"role":"assistant",...}],"rejected":[...]}` | T05, P13 |
| raw text (CPT) | `{"text":"..."}` | manuals, books, code for continued pre-training |

**Rules a valid chat example must obey** (enforced by `peftkit.data.format.validate`): roles only `system/user/assistant`; optional single system message first; then strictly alternating `user, assistant, user, assistant…`; it must **end with assistant**; no empty content.

**What `.jsonl` means:** one complete JSON object per line, no commas between lines, UTF-8. It streams (a 10 GB file never has to fit in memory) and appends safely. That is why every trainer uses it.

## 2. Chat templates: ChatML, Llama-3, Mistral

The model never sees JSON. The tokenizer's **chat template** (a Jinja program stored in `tokenizer_config.json` or `chat_template.jinja`) renders the messages into one string with special tokens. **You must train and serve with the model's own template.** Never write template tokens into your data by hand.

```powershell
python labs/lab8_chat_templates.py
```

**YOU SHOULD SEE (real output):**

```text
===== chatml        (Qwen, SmolLM, many others)
<|im_start|>system
You are a pump expert.<|im_end|>
<|im_start|>user
Max pressure?<|im_end|>
<|im_start|>assistant
6 bar.<|im_end|>
-- trained tokens: 20/91 (22%)
-- text the loss sees: '6 bar.<|im_end|>7.5 bar for 10 minutes.<|im_end|>'

===== llama3        (Llama 3.x)
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a pump expert.<|eot_id|><|start_header_id|>user<|end_header_id|>

Max pressure?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

6 bar.<|eot_id|>...
-- trained tokens: 38/297 (13%)

===== mistral       (Mistral-Instruct v0.x style: NO system role, it is glued to the first user turn)
<s>[INST] You are a pump expert.

Max pressure? [/INST] 6 bar.</s>[INST] And for testing? [/INST] 7.5 bar for 10 minutes.</s>
-- trained tokens: 27/92 (29%)
```

| Template | Turn markers | End-of-turn token (must be trained) | Gotcha |
|---|---|---|---|
| ChatML | `<\|im_start\|>role\n … <\|im_end\|>` | `<\|im_end\|>` | Qwen3 adds `<think>` blocks for reasoning mode; check `enable_thinking` in its template |
| Llama-3 | `<\|start_header_id\|>role<\|end_header_id\|>\n\n … <\|eot_id\|>` | `<\|eot_id\|>` | `eos_token` in some configs is `<\|end_of_text\|>`, but the chat turn ends with `<\|eot_id\|>` |
| Mistral | `[INST] … [/INST] answer</s>` | `</s>` | no system role; newer Mistral templates differ by version: always print yours |

**Print the template of YOUR model** (do this once for every new model):

```powershell
python -c "from transformers import AutoTokenizer as T; t=T.from_pretrained('MODEL_ID'); print(t.chat_template[:1500])"
```

## 3. Loss masking: why it matters, what breaks without it

Training computes a loss at every position whose **label** is not `-100`. **Assistant-only loss** sets the labels of system and user tokens to `-100`, so only the answer (and its end-of-turn token) is learned.

```powershell
python labs/lab9_loss_masking.py
python -m peftkit.tokenize_inspect --tokenizer outputs/tiny --file data/lesson/train.jsonl --index 0 --limit 40
```

**YOU SHOULD SEE (real output):**

```text
untrained model          answer NLL = 5.993
loss on ALL tokens   reported train loss = 5.670   answer NLL = 5.802
assistant-only loss  reported train loss = 5.435   answer NLL = 5.050
Reported train losses are NOT comparable: they average over different token sets.
...
total 62 tokens, trained on 8 (13%), EOS trained: True
```

Same data, same steps: assistant-only loss learned the **answer** much better (NLL 5.05 vs 5.80). The capacity was not wasted on learning to write questions.

| What breaks | Cause | Symptom | Check |
|---|---|---|---|
| model writes the user's next question | loss on user tokens (no masking) | answers followed by "User: …" | `tokenize_inspect` trained % near 100% |
| model never stops | end-of-turn token not in the trained span (or pad = eos and masked) | rambles to `max_tokens` | `EOS trained: False` |
| model learns nothing | everything masked (template without `{% generation %}` markers + assistant_only_loss) | flat loss | trained % = 0 |
| answer cut off | `max_seq_length` shorter than the example; truncation cuts from the END | truncated answers | doctor's truncation check, p95 length |
| multi-turn only learns the last answer | only last turn unmasked | turn 2+ quality poor | every assistant turn must be trained (lab 52) |

**How the toolkit does it:** `data.assistant_only_loss: true` in the YAML → TRL builds the mask from the template's `{% generation %}` … `{% endgeneration %}` markers. Many official templates **do not have these markers**. The doctor (`peftkit.diagnostics`) FAILs if you ask for assistant-only loss on such a template, and `peftkit.data.templates` has marker-equipped ChatML/Llama-3/Mistral versions. If you cannot use markers, the fallback is `assistant_only_loss: false` (loss on everything). It works, it is just less efficient, and you must still see `EOS trained: True`.

## 4. Deduplication and filtering pipelines

Duplicates make the model memorise; near-duplicates inflate your eval score; contamination (eval items inside training) makes the eval meaningless. The toolkit's pipeline (`peftkit.data.quality`), in order:

```text
read rows
 -> exact dedup          SHA-1 of normalised text (NFKC, lower-case, punctuation removed)
 -> heuristic filters    answer too short | text too long | 3-gram repetition ratio > 0.5 | non-printable chars
                         | refusal boilerplate ("as an AI language model") | suspected prompt injection / malicious URL
 -> PII                  emails, phones, card numbers, IPv4 -> <EMAIL>, <PHONE>... (scrub) or drop
 -> contamination        any 13-gram shared with your eval/benchmark files -> drop
 -> near-dup             MinHash (64 permutations) + LSH (16 bands), confirmed with exact Jaccard >= 0.8
 -> write kept rows + a report counting every drop reason
```

```powershell
python -m peftkit.data.quality --inputs data/interim/all.jsonl --out data/interim/clean.jsonl --report reports/quality.json --near-dup 0.8 --pii scrub --contamination data/eval/test.jsonl
```

**Why MinHash + LSH:** exact Jaccard between all pairs is O(n²): 100k rows = 5 billion comparisons. MinHash compresses each row into 64 numbers whose agreement rate estimates Jaccard; LSH puts rows that agree on a whole band of 4 numbers into the same bucket, so only bucket-mates are compared. Roughly linear time.

## 5. Synthetic data generation (Self-Instruct, Alpaca-style)

**Self-Instruct** (Wang et al. 2022): start from ~175 human-written seed tasks → repeatedly show an LLM 6 human + 2 machine tasks → it proposes new instructions → keep only those with **ROUGE-L < 0.7** against every task in the pool (forces diversity), with length and banned-word filters → then generate an input/output for each new instruction. **Alpaca** (Stanford 2023) applied this to produce 52K examples from text-davinci-003 for LLaMA-7B.

The toolkit implements both, domain-configurable, against any OpenAI-compatible API (`peftkit.data.self_instruct`), plus **grounded** generation from your documents (`peftkit.data.synth`), an **LLM judge** filter with position-swap for pairwise (`peftkit.data.judge`), and preference-pair generation (`peftkit.data.preference`). Full commands: P07.

**Honest limits of LLM-generated data (read before you generate 100k rows):**

1. **It inherits the generator's errors and style.** Grounded generation (answer ONLY from this passage) + a judge filter + 30 rows read by a human is the minimum.
2. **Diversity collapses.** Without the ROUGE filter, after a few thousand rows the generator repeats itself. The toolkit stops paying for calls when no new instruction survives for 20 rounds (`stalled`).
3. **Licences travel.** Alpaca is CC BY-NC (no commercial use). Some API terms restrict training competing models. Record `generator_model` in each row's lineage (`peftkit.data.lineage`).
4. **Model collapse.** Training repeatedly on your own model's outputs drifts the distribution (lab 10, lab 41): keep real/human data in every round.
5. **Your eval set must NOT be generated by the same generator** from the same documents, or you are measuring how well the model copies the generator.

---

## CHECKPOINT

- [ ] I can write one valid `messages` example by hand and say why it is valid.
- [ ] I printed my model's chat template and know its end-of-turn token.
- [ ] `tokenize_inspect` on one of my rows shows 2-90% trained and `EOS trained: True`.
- [ ] I know the 6 stages of the cleaning pipeline and which flag controls each.
- [ ] I know the five limits of LLM-generated data.
