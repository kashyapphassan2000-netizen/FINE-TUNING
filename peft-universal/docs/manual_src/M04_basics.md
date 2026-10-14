# LAB MANUAL — PART 4: BASICS — Fine-tune a tiny model on your laptop, and prove it learned

:::why Concept (what, why, when)
You will run the complete loop — data, inspection, pre-flight check, baseline, training, evaluation, merge — on a tiny model that runs on any laptop CPU in seconds. **Why:** every later level uses exactly these commands on bigger models; learning them where mistakes cost nothing is the cheapest training you will ever get. **When:** before renting any GPU, and again whenever something on the GPU behaves strangely.
:::

All commands below were executed while building this manual; the "You should see" blocks are the real output.

## SECTION 1 — Make the tiny model

**WHERE:** PowerShell, project folder, `(.venv)` active

**GOAL:** a small offline model in `outputs\tiny`

Step 1. Type and press Enter:

```laptop
python scripts/make_tiny_model.py
```

**You should see:**

```out
tiny model ready: C:\Users\YOUR_NAME\code\peft-universal\outputs\tiny
```

:::check
- The folder `outputs\tiny` exists and contains `config.json` and `tokenizer.json`.
:::

## SECTION 2 — Read the training data

**WHERE:** VS Code

**GOAL:** understand what one training example looks like

Step 1. In the VS Code Explorer, open `data\lesson\train.jsonl`.

Step 2. Read the first line. It is one conversation:

```json one line of data/lesson/train.jsonl
{"messages": [{"role": "system", "content": "You are a pump support expert."},
              {"role": "user", "content": "What is the maximum pressure?"},
              {"role": "assistant", "content": "The maximum pressure is 6 bar."}]}
```

Step 3. Open `data\lesson\test.jsonl`. These are the **held-out** questions, phrased differently from training, with the expected answer.

:::warn Watch out
Never copy a test question into training. The whole point of the test set is that the model has not seen it.
:::

:::check
- You can say which file trains the model and which file judges it.
:::

## SECTION 3 — See the example exactly as the model sees it

**WHERE:** PowerShell

**GOAL:** confirm only the answer is trained, and the end-of-turn token is included

Step 1. Type and press Enter:

```laptop
python -m peftkit.tokenize_inspect --tokenizer outputs/tiny --file data/lesson/train.jsonl --index 0 --limit 40
```

**You should see (last line):**

```out
total 62 tokens, trained on 8 (13%), EOS trained: True
```

The rows above it show every token; the prompt tokens have `label -100` (not trained).

|If you see…|It means…|Do this next|
|---|---|---|
|trained on 0 (0%)|everything is masked; the model would learn nothing|check the chat template has generation markers (Reference chapter 43)|
|trained on ~100%|nothing is masked; it would learn to write questions|set `assistant_only_loss: true` in the config|
|EOS trained: False|it will never learn to stop|fix the template / pad token (Reference chapter 22)|

:::check
- Trained fraction between 2% and 90%, and `EOS trained: True`.
:::

## SECTION 4 — Run the pre-flight doctor

**WHERE:** PowerShell

**GOAL:** zero FAIL before any training

Step 1. Type and press Enter:

```laptop
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/lora.yaml configs/lesson.yaml
```

**You should see (real output):**

```out
[WARN] dataset size: 48 training examples
[WARN] duplicates in train: 44 exact duplicates
[PASS] train/eval leakage: 0/2 eval examples also appear in train
[PASS] truncation: 0.0% of examples exceed max_seq_length=128 (p95=88)
[PASS] chat template: present
[PASS] loss mask: 12.9% of tokens are trained on
[PASS] EOS is trained: end-of-turn token in the trained span: True
[PASS] trainable params: 32,768
[WARN] learning rate: 0.005 (typical 5e-05-0.001)
[PASS] optimizer steps: 120 planned steps
[PASS] effective batch: 8 sequences per update
11 checks: 8 pass, 3 warn, 0 fail
```

The three WARNs are expected for this toy (tiny, repeated data; a high learning rate a random-weight model needs). On your real data they must be read and fixed.

:::check
- `0 fail`. If not: each FAIL line has a `fix:` line under it — do exactly that.
:::

## SECTION 5 — Measure the baseline BEFORE training

**WHERE:** PowerShell

**GOAL:** the score to beat

Step 1. Type and press Enter:

```laptop
python -m peftkit.evaluate --model outputs/tiny --suite domain=data/lesson/test.jsonl --max-new-tokens 16 --out reports/lesson_base.json
```

**You should see:**

```out
{
  "domain": {
    "n": 2,
    "score": 0.0,
    "nll": 5.995332717895508
  }
}
```

:::check
- `reports\lesson_base.json` exists. Without a baseline, "it improved" cannot be measured.
:::

## SECTION 6 — Train

**WHERE:** PowerShell

**GOAL:** a trained LoRA adapter in `outputs\lesson\adapter`

Step 1. Type and press Enter:

```laptop
python -m peftkit.train --config configs/base.yaml configs/methods/lora.yaml configs/lesson.yaml
```

**You should see (last line):**

```out
{"method": "lora", "trainable_params": 32768, "global_step": 120, "train_loss": 4.959970219930013}
```

:::tip Under the hood (standard libraries)
`peftkit.train` builds `peft.LoraConfig(r=16, lora_alpha=32, target_modules=...)` and `trl.SFTConfig(assistant_only_loss=True, max_length=128, learning_rate=5e-3, ...)`, then runs `trl.SFTTrainer(model, args, train_dataset, eval_dataset, processing_class=tokenizer, peft_config=lora).train()`. The complete standalone version is `examples/standard/train_sft_raw.py` (Part 5B).
:::

:::warn Watch out
`trainable_params` must not be 0. If it is, the adapter did not attach — check `target_modules` (Reference chapter 88).
:::

:::check
- `outputs\lesson\adapter\adapter_model.safetensors` exists.
:::

## SECTION 7 — Evaluate and compare

**WHERE:** PowerShell

**GOAL:** a PASS verdict on held-out questions

Step 1. Type and press Enter:

```laptop
python -m peftkit.evaluate --model outputs/lesson/adapter --suite domain=data/lesson/test.jsonl --max-new-tokens 16 --out reports/lesson_tuned.json
```

**You should see:** `"score": 0.5` (it was 0.0).

Step 2. Type and press Enter:

```laptop
python -m peftkit.evaluate --compare reports/lesson_base.json reports/lesson_tuned.json
```

**You should see:**

```out
|suite|before|after|delta|
|---|---|---|---|
|domain|0.0|0.5|+0.500|

verdict: PASS: domain improved, no large general regression  regressions: 0
```

:::check
- Verdict PASS. (On a real project you also add a `general=` suite so forgetting is measured.)
:::

## SECTION 8 — Merge the adapter into the model

**WHERE:** PowerShell

**GOAL:** one normal model folder, proven identical to base + adapter

Step 1. Type and press Enter:

```laptop
python -m peftkit.merge --adapter outputs/lesson/adapter --out outputs/lesson/merged --dtype float32
```

**You should see:**

```out
{"base": "outputs/tiny", "adapter": "outputs/lesson/adapter", "peft_type": "LORA", "use_dora": false,
 "max_abs_logit_diff": 5.066394805908203e-07, "out": "outputs/lesson/merged"}
```

`max_abs_logit_diff` ≈ 5e-07 means the merged model gives the same outputs as base + adapter.

:::tip Under the hood (standard libraries)
`PeftModel.from_pretrained(base, adapter).merge_and_unload()`, then `save_pretrained()`. Standalone: `examples/standard/merge_publish_raw.py` (Part 5B) — its own run gave a difference of 5.0e-07.
:::

:::check
- The difference is below 1e-4. If it is large: you merged onto a different base than you trained on.
:::

## 4.9 Common mistake at this level

**Trusting the training loss.** The loss here went from ~5.9 to ~4.9 — that number alone says nothing about whether the model answers new questions. The only evidence is Section 7: the same held-out test, before and after. **How to spot it:** if your "result" slide shows a loss curve and no before/after table, you have not measured anything yet.
