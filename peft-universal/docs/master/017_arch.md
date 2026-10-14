# 17. Architecture

## 17.1 What runs where

```text Data flow
 LAPTOP (Windows, CPU)                      GPU VM #1  (Hyperstack, training session)
 ─────────────────────                      ───────────────────────────────────────────
 VS Code ──Remote-SSH──────────────────────▶ repo checkout
 pytest (CPU, tiny model)                    .venv-train (torch 2.14, peft, trl, bnb)
 data build (API calls only) ──WinSCP/git──▶ data/processed/*.jsonl
                                             peftkit.train  → outputs/<run>/adapter
                                             peftkit.merge  → outputs/<run>/merged
                                             peftkit.publish ──▶ Hugging Face Hub (private)
                                                               ──▶ Google Drive (rclone) / S3
                                             VM deleted (stop paying)

 GPU VM #2..N  (serving session)              USERS
 ────────────────────────────────             ─────
 .venv-serve (vllm 0.29.0, torch 2.13)        client ──HTTPS──▶ gateway (API keys, rate limit)
 vllm serve <hub-id>  :8000  ◀──────────────────────────────────┘      │
 /metrics (queue depth) ─────▶ Prometheus ─▶ KEDA or scaler.py ──▶ add/remove GPU nodes
```

## 17.2 Environments

|Env|Where|Holds|Why separate|
|---|---|---|---|
|laptop `.venv`|Windows|CPU torch + train stack + pytest|fast tests, no GPU bill|
|`.venv-train`|GPU VM|torch 2.14 CUDA 13, peft, trl, bnb|training + merging|
|`.venv-serve`|GPU VM|vllm 0.29.0 → torch 2.13|vLLM pins torch|
|`.venv-dd` (optional)|any|DataDreamer 0.46|pins old datasets/hub/numpy|

## 17.3 Time and money, honestly

This guide does not quote GPU prices: they change and differ by region and contract. Read them in the Hyperstack console or with the Pricebook API before you start, and write them in your own notes. What drives cost:

- **Idle time.** Billing starts when a VM is ACTIVE (Hyperstack docs). Download models to fast local disk once, keep a `tmux` session, and delete or hibernate the VM when training ends.
- **Tokens trained.** Training time is roughly proportional to (examples × average tokens × epochs). Measure your own tokens/second in a 20-step dry run (chapter 56) and multiply; do not trust anyone's table, including this one.
- **Synthetic data API calls.** `peftkit.data.*` prints token usage per stage so you can price it with your provider's rates.

Measured here: the whole CPU test suite takes about 15 seconds on one CPU core.

## 17.4 Honest limitations

- PEFT (the Hugging Face library) is PyTorch-only. The math is framework-independent (lab 7 shows LoRA in JAX), but this template trains with PyTorch. For JAX/TPU, the same YAML ideas apply with a JAX stack such as EasyDeL or MaxText; that is out of scope here.
- vLLM 0.29.0 cannot serve DoRA or AdaLoRA adapters directly. The template merges them (exact, tested).
- QLoRA with FSDP sharding is not wired in. QLoRA runs on one GPU or with DDP (one 4-bit copy per GPU). For models too large for one GPU even in 4-bit, use LoRA + FSDP2 on a bf16 base.
- The GPU memory estimator is a planning tool. It matched exact parameter counts in tests, but activation memory is a formula, not a measurement.
- Only causal (decoder-only) text models are covered. Vision-language models need a processor and different collators.
