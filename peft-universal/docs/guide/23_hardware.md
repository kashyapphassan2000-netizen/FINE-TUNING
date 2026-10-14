# 23. Hardware sizing: the universal "which GPU?" template

:::why The memory equation
$$ VRAM ≈ base weights + adapter state + activations + logits + overhead
- base weights: params × 2 bytes (bf16) or ≈ 0.52 byte (NF4 + double quant), plus fp32 non-quantized tensors for QLoRA
- adapter state: trainable × 16 bytes (fp32 weight 4 + grad 4 + Adam 8), or × 10 with 8-bit Adam
- activations: with gradient checkpointing ≈ layers × 2 × seq × batch × hidden bytes + one full layer
- logits: batch × seq × vocab × 4 bytes (TRL's chunked loss reduces this)
- serving adds the KV cache: 2 × layers × kv_heads × head_dim × 2 bytes per token
:::

## 23.1 The estimator

@file src/peftkit/hardware.py

It reads only `config.json` (from the Hub or a local folder). Tests prove the parameter count equals the real model's, and the adapter-parameter formula equals PEFT's count for LoRA (16,384), DoRA (17,408) and AdaLoRA (24,744) on the tiny model. Llama-3.1-8B comes out at 8.03B and 70B at 70.55B parameters, matching their published sizes.

```any
python -m peftkit.hardware meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --batch 2
python -m peftkit.hardware configs/models/llama-3.1-70b.config.json      # offline copy of the config
```

```out Real output for the 8B config (seq 2048, batch 1, r 16)
method     weights    trainable   optim    acts  logits   TOTAL  fits on
lora         14.96   41,943,040    0.62    0.84    0.98   20.64  1x RTX 4090 / L4 / A10G (24 GB total)
qlora         7.27   41,943,040    0.39    0.84    0.98   11.92  1x RTX 4090 / L4 / A10G (24 GB total)
dora         14.96   43,319,296    0.65    0.94    0.98   20.77  1x RTX 4090 / L4 / A10G (24 GB total)
SERVING (vLLM, bf16, max_len=8192, 16 full-length seqs): weights 14.96 GB, KV 0.125 MB/token, KV 16.0 GB, total 35.55 GB
```

```out Real output for the 70B config
lora        131.42  207,093,760    3.09    3.17    0.98  154.02  1x B200 (180 GB total)
qlora        40.72  207,093,760    1.93    3.17    0.98   52.98  1x A100 80GB / H100 80GB (80 GB total)
SERVING ... total_gb 190.06 → fits on: 2x H200 (282 GB total)  (tensor-parallel size = GPU count)
```

:::warn How to read it
- 20.6 GB on a 24 GB card is tight. Real runs add allocator fragmentation and longer sequences. Prefer a card with 20-30% headroom, or use QLoRA.
- Multi-GPU training rows assume FSDP sharding. DDP (plain data parallel) puts a full copy on every GPU and does **not** reduce per-GPU memory.
- Always confirm with a 20-step dry run: `run_meta.json` records `peak_vram_gb` from `torch.cuda.max_memory_allocated()`.
:::

## 23.2 Decision procedure (any model)

1. `python -m peftkit.inspect_model <model>` → architecture, parameter count, linear layer names.
2. `python -m peftkit.hardware <model> --seq-len <p95 of your data> --batch <b>` → pick the smallest setup with headroom.
3. Check GPU stock in the Hyperstack console for that flavor and region.
4. Rent it, run preflight (24.4), run a 20-step dry run, read `peak_vram_gb`, then decide batch size.
5. For serving, size the KV cache for your concurrency and context length; vLLM takes `--gpu-memory-utilization` of the card and uses what is left after weights for KV cache.

## 23.3 Other resources

|Resource|Guideline|
|---|---|
|System RAM|≥ model size in bf16 for loading (plus headroom); more for data processing|
|Disk|base model ×2 (download + merged copy) + checkpoints + datasets; use the VM's ephemeral NVMe for HF cache|
|CPU cores|4+ per GPU for dataloaders and tokenisation|
|Interconnect|NVLink helps tensor parallel and FSDP; PCIe-only cards are fine for single-GPU and DDP|
