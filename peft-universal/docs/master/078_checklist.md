# 78. Master checklist and cheat sheet

## 78.1 One page, pinned above the desk

```text BEFORE YOU TRAIN
  compat      python -m peftkit.compat <model>                  -> verdict, target names, tokenizer facts
  plan        python -m peftkit.plan <model> --seq-len <p95>     -> GPU, memory, KV cache, cost inputs
  data        build_dataset.sh -> quality --contamination        -> dedup, PII, leakage, judge
  tokens      tokenize_inspect --file train.jsonl                -> mask 2-90%, EOS trained, no truncation
  doctor      python -m peftkit.diagnostics --config ...         -> zero FAIL
  baseline    evaluate --model <base> --suite domain= general=   -> the number to beat
  dry run     train --set train.max_steps=20                     -> peak VRAM, tokens/sec, cost

WHILE TRAINING
  watch       train loss falling; eval every 20-50 steps; best-eval checkpoint kept
  stop        when eval stops improving, not when the epochs run out

AFTER TRAINING
  evaluate    evaluate --model <adapter> ... ; evaluate --compare base.json tuned.json
  verdict     domain up? general not down? regressions reviewed item by item?
  analyse     label 30 failures by cause; fix the biggest bucket; re-run
  merge       merge --adapter ... --base <hub id>   -> max_abs_logit_diff tiny
  serve       serve_check -> vllm serve -> gateway -> one real answer with an API key
  record      run_meta.json + one paragraph: before/after, cost, limits
```

## 78.2 Numbers worth memorising

|Thing|Value|
|---|---|
|LoRA learning rate|1e-4 to 3e-4 (2e-5 is a full-fine-tuning default and too low)|
|rank / alpha start|16 / 32|
|effective batch|16-64 sequences|
|warmup|3% of steps (`warmup_steps: 0.03`)|
|epochs|1-3, stop on eval|
|Adam memory|~16 bytes per trainable parameter (10 with 8-bit)|
|bf16 weights|2 bytes per parameter; NF4 ≈ 0.52|
|trained-token fraction|2-90%, otherwise the mask is wrong|
|eval set|100-300 human-verified items, never in training|
|general suite|≥ 50 items outside your domain|
|failure analysis batch|30-50 items before changing anything|

## 78.3 The ten commandments of not fooling yourself

1. Build the test set before you build the training set.
2. Measure the base model first.
3. Never evaluate on anything the model trained on.
4. Loss is not a result; a task score on unseen data is.
5. Measure a general suite, or you will ship a model that forgot how to talk.
6. Change one thing per run, and write the result down.
7. Read 30 rows of your own data before blaming the model.
8. Dry-run before you pay for hours.
9. Delete the GPU when you are done.
10. If you cannot reproduce it from the repo, it did not happen.

## 78.4 Verification of this volume

```any Everything in both volumes, on a laptop, no GPU needed
python -m pytest -q
```

```out REAL OUTPUT (this container, CPU only)
55 passed, 1 skipped
```

The one skip is the QLoRA test, which needs an NVIDIA GPU. The 13 labs of this volume (11-23) are separate programs you can run the same way:

```any
python labs/lab11_training_mechanics.py       # ... through lab23_dpo.py
```

:::check You have finished this book when
- You can hand a colleague an unseen model name and your repo, and they can follow chapter 75 without asking you anything.
- You have at least one model in production with a measured before/after number.
- You have your own version of the debugging table in chapter 54, with signatures you found yourself.
:::
