# 35. Compute, time and cost for any model

## 35.1 Counting tokens first

Everything starts from the number of tokens you will train on:

$$ tokens = examples × average tokens per example × epochs

```any Measure it exactly, do not guess
python -m peftkit.data.format --inputs data/processed/train.jsonl --out-dir /tmp/count --tokenizer <your base model>
python - <<'PY'
from transformers import AutoTokenizer; import json
tok = AutoTokenizer.from_pretrained("<your base model>")
n = [len(tok.apply_chat_template(json.loads(l)["messages"], tokenize=True, return_dict=True)["input_ids"])
     for l in open("data/processed/train.jsonl")]
print("examples", len(n), "total tokens", sum(n), "mean", sum(n)//len(n), "p95", sorted(n)[int(.95*len(n))])
PY
```

The p95 is the number to use for `model.max_seq_length`; the total is what drives cost.

## 35.2 The FLOPs law

For full training the classic estimate is 6ND (N = parameters, D = tokens): 2 forward, 4 backward. With PEFT the base weights are frozen, so no weight gradients are computed for them:

$$ FLOPs ≈ 4 × N_{active} × D    (no gradient checkpointing)
$$ FLOPs ≈ 6 × N_{active} × D    (with gradient checkpointing: one extra forward)

**Use N_{active}, not N_{total}.** For MoE models only the selected experts do work per token: Kimi-K2-like = 32.9B active out of 1026B. That is why a trillion-parameter MoE costs about as much *compute* per token as a 33B dense model, while still needing the *memory* of a trillion.

## 35.3 From FLOPs to hours to money

$$ tokens/sec per GPU = MFU × GPU peak FLOPs/sec / (6 × N_{active})
$$ GPU-hours = D / (tokens/sec per GPU) / 3600
$$ wall-clock hours = GPU-hours / number of GPUs
$$ cost = GPU-hours × price per GPU-hour

MFU ("model FLOPs utilisation") is the fraction of the GPU's theoretical peak you actually achieve. It depends on model, sequence length, kernels and interconnect. **Measure it; do not trust a number from a blog.** The planner accepts either a measured throughput or a peak-FLOPs + MFU pair:

```any Measured (preferred): take tokens/sec from your 20-step dry run
python -m peftkit.plan <model> --dataset-tokens 5e7 --epochs 2 --tokens-per-sec <measured> --price-per-gpu-hour <price>
# Estimated (before you have a machine): supply your GPU's peak bf16 FLOPs from its spec sheet
python -m peftkit.plan <model> --dataset-tokens 5e7 --gpu-flops <e.g. 312e12 for A100> --mfu 0.3
```

```any Get tokens/sec from a dry run
python - <<'PY'
import json; m = json.load(open("outputs/dryrun/run_meta.json"))
steps, secs = m["global_step"], m["train_seconds"]
cfg = m["config"]["train"]; tokens_per_step = cfg["per_device_batch_size"] * cfg["grad_accum"] * <avg_tokens>
print("sec/step", secs/steps, "tokens/sec/GPU", tokens_per_step/(secs/steps))
PY
```

:::math Worked example (arithmetic only, no invented prices)
20M tokens, 2 epochs = 40M tokens. Measured 1,500 tokens/sec on one GPU.
GPU-hours = 40,000,000 / 1,500 / 3600 = 7.4 hours on one GPU.
On 4 GPUs (DDP, near-linear if the data pipeline keeps up): ~1.9 hours wall-clock, still 7.4 GPU-hours of cost.
Cost = 7.4 × (your price per GPU-hour). Add the idle minutes for setup and download.
:::

## 35.4 Serving cost

Serving is billed by GPU-time, not by request, so the cost per request depends entirely on how busy you keep the GPU.

$$ cost per 1M output tokens ≈ (GPUs × price per GPU-hour) / (tokens/sec × 3600) × 1,000,000

Decode speed is limited mostly by memory bandwidth (every generated token reads the active weights), so: smaller active weights (MoE, quantization) = faster decode; bigger batches = far better cost per token, at some latency cost. Measure your own throughput with a load test before you promise anyone a price.

## 35.5 Sixteen ways to spend less (all available in this template)

|Hack|Effect|Where|
|---|---|---|
|dry run before the long run|stops a 10-hour OOM crash|ch 26|
|`packing: true`|fills every sequence, fewer padding tokens|`data.packing`|
|`max_seq_length` = p95 of your data, not 8192|memory and time scale with sequence length|35.1|
|QLoRA on dense models|~4x smaller weights|ch 5|
|attention-only targets on MoE|190x fewer trainable params (measured)|ch 33.4|
|gradient checkpointing|much less activation memory, ~30% slower|`train.gradient_checkpointing`|
|8-bit / paged optimizer|less optimizer memory, survives spikes|`train.optim`|
|TRL `chunked_nll` loss (default)|large-vocab logits no longer dominate|ch 26|
|Liger kernels (`use_liger_kernel` in SFTConfig)|fused kernels, less memory; verify with your versions|TRL|
|flash-attention kernels|faster attention, less memory|`model.attn_implementation`|
|fewer epochs + early stop on eval|most gains come in epoch 1|ch 9|
|smaller base model first|prove the pipeline on 135M|ch 34.3|
|pre-download weights to fast local disk / snapshot|no paid idle time downloading|ch 24|
|`tmux` + checkpoint + `--resume`|a crash costs minutes, not hours|ch 26|
|delete/hibernate the VM immediately|the biggest real-world waste|ch 27|
|one base + many adapters instead of many full models|one GPU serves all domains|ch 38|

:::warn What actually burns money for beginners
Not the training. It is: idle GPUs left running overnight, repeated downloads of a 100 GB model, and 10-hour runs on data nobody read first. Fix those three and your bill drops more than any kernel trick.
:::

:::check
- I know my dataset's total tokens and p95 length.
- I have a measured tokens/sec from a dry run.
- I multiplied it out and wrote the expected cost down BEFORE starting.
:::
