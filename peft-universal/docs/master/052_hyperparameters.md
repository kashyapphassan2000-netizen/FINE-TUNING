# 52. Hyperparameters, with a reason for every value

:::warn New in this edition
A running server is not a validated server. After this chapter, evaluate the SERVED model (chapter 53) and load test it (chapter 54) before the gateway and autoscaling in chapter 56.
:::


## 52.1 What each knob does

|Knob|What it controls|Start|Raise when|Lower when|
|---|---|---|---|---|
|learning rate|step size|2e-4 (LoRA/QLoRA), 1e-4 (DoRA, LoRA+ A)|loss barely moves|loss spikes or NaN|
|batch size|examples per forward|as large as memory allows|GPU is idle|OOM|
|gradient accumulation|virtual batch|4-8|you want a bigger effective batch without memory|you want faster feedback|
|epochs|passes over data|1-3|underfitting (ch 57)|eval loss rising|
|warmup|slow start to protect early steps|3% of steps|unstable start|very short runs|
|weight decay|pull weights toward zero|0.0-0.1|overfitting|underfitting|
|LR scheduler|shape of the decay|cosine|-|-|
|LoRA rank r|adapter capacity|16|underfitting, complex task|overfitting, tiny data|
|LoRA alpha|update scale (α/r)|2r|-|-|
|dropout|regularisation|0.05|overfitting|large data|
|max sequence length|tokens per example|p95 of your data|answers truncated|memory pressure|
|gradient clipping|caps exploding updates|1.0|-|-|

**Effective batch = per_device_batch_size × grad_accum × GPUs.** That is the number that actually matters for stability; 16-64 is a good range.

## 52.2 A real sweep (executed)

@file labs/lab15_hyperparameter_sweep.py
@lab lab15_hyperparameter_sweep

Read it honestly:

- **lr 2e-5** — the default in many trainers, built for full fine-tuning — produces 0.7% token accuracy. It is not "safe", it is broken for LoRA.
- **lr 2e-3** was best here; **5e-2** is past the edge and the score collapses.
- **Rank** 2 → 8 → 32 buys accuracy in decreasing steps while adapter size grows 7x. Diminishing returns are the normal shape.

These are numbers for a 120k-parameter random model on four examples. **They will not transfer to your model.** The method transfers: change one knob at a time, keep everything else fixed, compare on the same eval set.

## 52.3 A tuning order that wastes the least money

1. Fix the data first. No hyperparameter fixes bad data.
2. Get lr right (try 1e-4, 2e-4, 5e-4; one epoch each on a subset).
3. Set `max_seq_length` from the p95 of your token lengths.
4. Set effective batch to 16-64 within your memory budget.
5. Choose epochs by watching eval loss, not by picking a number.
6. Only then tune rank, then dropout/weight decay.
7. Only then try a different method (chapter 60).

:::check
- I know the effective batch of my run.
- My lr is in the 5e-5..1e-3 band unless I have a measured reason.
- I changed one thing at a time and recorded each result (ch 53).
:::
