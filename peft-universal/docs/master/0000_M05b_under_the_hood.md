# LAB MANUAL — PART 5B: Under the Hood — the same steps with standard libraries only

:::why Concept (what, why, when)
Every `peftkit` command in this manual is a thin layer over standard open-source libraries: **transformers**, **peft**, **trl**, **datasets**, **huggingface_hub**. This Part shows the same training, merge and publish steps written **only** with those libraries — no `peftkit` import anywhere. **Why:** if this repository ever disappears, you still have complete, working recipes, and you understand exactly what each command does. **When:** read it once after Part 5; come back whenever you need to customise something `peftkit` does not expose.
:::

## 5B.1 Map: peftkit command → the standard code underneath

|peftkit command|Standard library equivalent|Key objects|
|---|---|---|
|`peftkit.train`|`trl.SFTTrainer` + `SFTConfig` + `peft.LoraConfig`|`SFTConfig(assistant_only_loss=True, max_length=…)`, `LoraConfig(r, lora_alpha, target_modules)`|
|`--method qlora`|`transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`|passed as `quantization_config=` to `from_pretrained`|
|`peftkit.tokenize_inspect`|`tokenizer.apply_chat_template(..., return_dict=True, return_assistant_tokens_mask=True)`|the assistant mask the trainer uses|
|`peftkit.evaluate`|`model.generate` + your scoring function|temperature 0, same suites before/after|
|`peftkit.merge`|`PeftModel.from_pretrained(base, adapter).merge_and_unload()`|folds B·A·(α/r) into W|
|`peftkit.publish`|`model.push_to_hub(repo, private=True)` + `tokenizer.push_to_hub(...)`|needs a write token|
|`scripts/serve_vllm.sh`|`vllm serve MODEL --served-model-name NAME --max-model-len N --api-key KEY`|OpenAI-compatible server|

## SECTION 1 — Train with transformers + peft + trl only

**WHERE:** PowerShell or WSL, project folder, environment active

**GOAL:** the Part 4 run, reproduced without peftkit

@file examples/standard/train_sft_raw.py

Step 1. Type and press Enter:

```any
python examples/standard/train_sft_raw.py --model outputs/tiny --train data/lesson/train.jsonl --eval data/lesson/eval.jsonl --out outputs/raw_lesson --epochs 20 --lr 5e-3
```

**You should see (real output, last line):**

```out
{'train_loss': 4.8823, 'steps': 120, 'saved': 'outputs/raw_lesson/adapter'}
```

Step 2. Score it with the same held-out test as Part 4:

```any
python -m peftkit.evaluate --model outputs/raw_lesson/adapter --suite domain=data/lesson/test.jsonl --max-new-tokens 16
```

**You should see (real):** `"score": 1.0` — the standard-library run learned the task too (Part 4's run scored 0.5; with two test items and a random-weight toy, a difference this size is noise).

|Line in the script|What it does in plain words|
|---|---|
|`LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")`|attach rank-16 LoRA to every Linear layer except the output head|
|`SFTConfig(assistant_only_loss=True)`|train only on the assistant's replies (the chat template marks them)|
|`warmup_steps=0.03`|in transformers 5 `warmup_ratio` was removed; a float here is read as a ratio|
|`max_length=…`|sequences longer than this are truncated — set it to your p95 length|
|`SFTTrainer(..., peft_config=lora)`|the trainer wraps the model with LoRA for you and freezes the base|
|`trainer.save_model(".../adapter")`|saves only the adapter (megabytes), not the base model|

:::check
- `outputs/raw_lesson/adapter/adapter_model.safetensors` exists and the held-out score rose above the baseline.
:::

## SECTION 2 — Merge and publish with peft + huggingface_hub only

**WHERE:** same terminal

**GOAL:** a merged model, proven exact, optionally on the Hub

@file examples/standard/merge_publish_raw.py

Step 1. Type and press Enter:

```any
python examples/standard/merge_publish_raw.py --base outputs/tiny --adapter outputs/raw_lesson/adapter --out outputs/raw_lesson/merged
```

**You should see (real):**

```out
max |logit difference| after merge: 4.991888999938965e-07
saved merged model to outputs/raw_lesson/merged
```

Step 2. To publish, add `--hub HF_USER/REPO --private` (needs `HF_TOKEN` with write access).

:::warn Watch out: a real transformers 5 trap (hit while writing this Part)
`tokenizer.apply_chat_template(..., return_tensors="pt")` returns a **BatchEncoding** (a dict-like object) in transformers 5, not a tensor. Passing it straight to the model fails with `embedding(): argument 'indices' must be Tensor, not BatchEncoding`. Ask for `return_dict=True` and take `["input_ids"]`, as the script does.
:::

:::check
- The logit difference is below 1e-4.
:::

## SECTION 3 — Serve with the vLLM command directly

**WHERE:** GPU VM, `.venv-serve`

**GOAL:** the same server `scripts/serve_vllm.sh` starts, typed by hand

```gpu
vllm serve HF_USER/REPO --served-model-name SERVED_NAME --max-model-len 8192 --api-key $VLLM_API_KEY --host 0.0.0.0 --port 8000
```

For LoRA adapters served on one base model add `--enable-lora --lora-modules NAME=path/to/adapter --max-lora-rank 16` (Reference chapter 71).

## 5B.4 Common mistake at this level

**Copying a blog's `target_modules` list.** `["q_proj", "v_proj"]` exists in Llama-style models but not in BERT (`query`, `value`) or T5 (`q`, `v`). With a wrong list, peft raises "Target modules not found" — or, worse, matches fewer layers than you think. **How to spot it:** call `print_trainable_parameters()` (the raw script does) and compare with `peftkit.plan`'s "trainable LoRA params" line. Use `"all-linear"` or discovered names (Reference chapter 88).
