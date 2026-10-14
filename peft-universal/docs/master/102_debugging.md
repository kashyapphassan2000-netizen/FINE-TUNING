# 102. Universal debugging

@file labs/lab61_universal_debugging.py
@lab lab61_universal_debugging

## 102.1 The table, for any architecture

Format: SYMPTOM → LIKELY CAUSES → DIAGNOSTIC COMMAND → EXPECTED RESULT → FIX → VERIFY.

|Symptom|Likely causes|Diagnostic|Expected when healthy|Fix|Verify|
|---|---|---|---|---|---|
|NaN loss|lr too high, fp16 overflow, bad labels, log(0)|`loop.train` stops at the step; `trace_forward` std/NaN column|finite loss, stable std per layer|lower lr, `clip=1.0`, bf16, fix labels|10 steps finite|
|loss not decreasing|nothing trainable, lr too low, wrong labels/mask, adapters unused|`gradient_report`; overfit one batch|every adapter has a gradient; one-batch loss falls|attach/unfreeze correctly, lr 1e-4..1e-3|one-batch overfit works|
|loss down, eval worse|overfitting, leakage in train, eval mismatch|compare train vs eval curves; contamination check (ch 45)|both fall|fewer epochs, more data, lower rank|eval at best checkpoint|
|zero gradients|frozen by mistake, detached tensor, `torch.no_grad` in forward|`gradient_report` → no_or_zero_grad|empty list|remove the detach / no_grad, unfreeze|report ok|
|exploding gradients|lr, no clipping, bad normalisation|log `grad_norm` from the loop history|bounded|clip, lower lr, warmup|grad_norm stable|
|vanishing gradients|very deep stack without residuals, saturated activations|per-layer grad norms from `trace_forward(backward_loss=...)`|similar order of magnitude across layers|residuals, norms, adapter placement|norms comparable|
|OOM|batch, sequence/image size, full FT|`peftkit.plan` / dry run peak memory|fits with headroom|smaller batch + accumulation, checkpointing, QLoRA, bf16|dry run fits|
|CUDA device-side assert|label index out of range, bad token id|run on CPU once: the error becomes readable|—|fix label range / vocab|CPU run passes|
|wrong shape|collator, processor mismatch|`trace_forward(...).table()`; print batch shapes|shapes match forward's expectations|fix collator/processor|trace passes|
|wrong dtype / device|mixed fp32/bf16 inputs, CPU tensors to a CUDA model|print `x.dtype, x.device` in a forward hook|consistent|cast inputs; move batch to device|—|
|wrong labels|off-by-one shift, padding not -100, class ids off|decode a batch; `tokenize_inspect`|labels align with targets|fix label construction|one-batch overfit|
|wrong masking|prompt trained, EOS masked|`tokenize_inspect` (text); mask sums|trained span = answer + EOS|template markers / masks|report sane|
|wrong chat template / processor / tokenizer|mismatch with the model's own files|load all from the SAME repo and revision|identical special tokens|reload consistently|served parity (ch 67)|
|LoRA targets not found|names from another architecture|`uadapt.targets.discover`|discovered full paths|use them|inject reports wrapped > 0|
|no trainable parameters|everything frozen|`param_counts(model)["trainable"]`|> 0|inject / unfreeze|loop starts|
|adapter trains nothing (silent)|adapter on an unused layer, or wrapper bypassed|`verify_called` + `gradient_report`|called or read, and receives gradients|adapt another layer; effective-weight wrapper|grads present|
|model outputs garbage|tokenizer/model mismatch, wrong merge base|generate with the adapter and with the merged model|identical|merge onto the exact base+revision|merge diff ~0|
|memorises the dataset|duplicates, tiny data, too many epochs|dedup report; train vs eval gap|small gap|dedup, more variety, early stop|eval on unseen entities|
|catastrophic forgetting|narrow data, high lr|general suite before/after (ch 59)|no large drop|replay data, fewer epochs|compare verdict PASS|
|inference / serving mismatch|template, dtype, quantization, adapter name, sampling|`eval_endpoint --offline` parity (ch 67)|gap ≤ 0.05|align serving config|parity PASS|

## 102.2 The discipline

Change one thing. Re-run the same diagnostic. Only then continue. Write every new failure signature into this table for your own stack.
