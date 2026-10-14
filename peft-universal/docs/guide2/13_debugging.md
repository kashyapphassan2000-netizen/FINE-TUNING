# 13. The debugging lab: break it on purpose

:::why Why break things deliberately
Every failure has a signature. If you have seen the signature once, you fix it in two minutes instead of two days.
:::

## 13.1 Failures produced on purpose, with their real messages

@file labs/lab20_debug_gallery.py
@lab lab20_debug_gallery

## 13.2 Symptom → diagnosis → root cause → fix → verification

|Symptom|Diagnosis command|Root cause|Fix|Verify|
|---|---|---|---|---|
|`Target modules ... not found`|`peftkit.inspect_model`|names from another architecture|use the printed names|`trainable params > 0`|
|loss flat from step 0|check `trainable` in the log|no adapter attached / lr 2e-5|attach adapter, lr 2e-4|loss falls within 20 steps|
|loss is `nan` immediately|`tokenize_inspect`|all labels -100, or fp16 overflow|fix the mask; use bf16|trained fraction 2-90%|
|loss goes to `nan` after N steps|log history|lr too high, no clipping|lower lr, `max_grad_norm: 1.0`|stable curve|
|`assistant_only_loss` error|`peftkit.diagnostics`|template without generation markers|peftkit templates, or turn it off|doctor PASS|
|model never stops|`tokenize_inspect`|EOS not in the trained span|fix template/pad token|EOS trained: True|
|model writes the user's next turn|`tokenize_inspect`|no loss masking|`assistant_only_loss: true`|trained text = answer only|
|answers are cut mid-sentence|doctor truncation check|`max_seq_length` below p95|raise it|truncation < 2%|
|great eval, useless in production|`data.quality --contamination`|leakage (ch 6)|clean split, re-measure|honest score|
|domain up, everything else down|`evaluate --compare`|catastrophic forgetting|replay data, fewer epochs|general delta ≈ 0|
|CUDA OOM|`peftkit.plan`|batch/seq/model too big|halve batch, QLoRA, bigger GPU|dry run fits|
|garbage output after merge|`merge` report|wrong base or revision|merge with `--base <hub id>`|`max_abs_logit_diff` tiny|
|vLLM refuses the adapter|`peftkit.serve_check`|DoRA/AdaLoRA/modules_to_save|merge and serve merged|server starts|

## 13.3 The doctor: run it before every training run

@file src/peftkit/diagnostics.py

```any
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/<m>.yaml
```

```out shape of the output
[PASS] dataset size: 4200 training examples
[FAIL] train/eval leakage: 37/200 eval examples also appear in train
         fix: remove them: your eval score is otherwise meaningless
[WARN] learning rate: 2e-05 (typical 5e-05-0.001)
         fix: 2e-5 is a FULL fine-tuning default and is too low for LoRA
...
14 checks: 11 pass, 2 warn, 1 fail
```

It exits non-zero on any FAIL, so `peftkit.diagnostics && peftkit.train` is a safe habit and a good CI gate.

## 13.4 Debugging method

1. **Reproduce small**: 50 examples, 20 steps, tiny model. Most bugs reproduce there in seconds.
2. **Change one thing.**
3. **Look at the data the model actually sees** (`tokenize_inspect`) before theorising about the model.
4. **Bisect**: base model + your prompt (is it the data?), your data + a known-good config (is it the config?).
5. **Write down the signature** when you fix it. That is how you build the table above for your own stack.

:::check
- Doctor exits 0 on my config.
- I have reproduced at least three of the gallery failures myself.
:::
