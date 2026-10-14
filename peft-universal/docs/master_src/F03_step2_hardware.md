# 5. Step 2 — How much hardware: to LOAD and to FINE-TUNE

:::why Two different questions
"Will it load?" is about weights + a little overhead. "Can I fine-tune it?" adds gradients, optimizer state and activations. People size for the first and get an OOM on the second.
:::

## 5.1 The arithmetic (do this before renting anything)

```text
LOAD (inference):      bytes/param x params  + KV cache + ~1.5 GB workspace
                       bf16 = 2 B/param, fp8 = 1, NF4 ~ 0.55
FINE-TUNE:             weights + 16 B x TRAINABLE params (Adam) + activations + logits, x1.1 overhead
                       LoRA/QLoRA: "trainable" is the ADAPTER, not the model - that is the whole saving
GPUS = ceil(total / (VRAM x 0.9))     (sharded: FSDP/ZeRO-3; plain DDP replicates the model per GPU)
```

## 5.2 The commands

```any
python -m peftkit.plan <model> --seq-len 4096 --batch 1          # per-method memory, GPU count, node layout
python -m peftkit.kv   <model> --context 32768 --gpu-gb 48       # KV cache: how many users fit
python -m peftkit.capacity --model <model> --users 2000 --active-share 0.1 \
    --requests-per-active-user-per-min 2 --prompt-tokens 600 --output-tokens 250 --latency-s 4 \
    --prices configs/providers.yaml                              # train GPUs, serve replicas, cost
```

## 5.3 Worked, for an 8B model (numbers from the deep chapters)

|Question|Answer|
|---|---|
|load in bf16|~15 GB weights → one 24 GB GPU, with room for a small KV cache|
|fine-tune, bf16 LoRA, seq 4096|~20.6 GB → 24 GB is tight, 48 GB comfortable|
|fine-tune, QLoRA (NF4)|~11.9 GB → one 24 GB GPU|
|full fine-tune with Adam|~128 GB → multiple GPUs, sharded|
|serve 2,000 users at 4 s latency|Little's law → 26.7 concurrent → replicas = measured capacity ÷ that|

## 5.4 The rule that saves money

Size from the **dry run**, not the formula: 20 steps on the real model prints peak VRAM and tokens/sec. The formula picks the GPU; the dry run confirms it before you commit hours.

:::check
- I know the GPU type, count, and whether one replica needs tensor parallelism.
- I know the training time and cost estimate, with my own price.
:::
