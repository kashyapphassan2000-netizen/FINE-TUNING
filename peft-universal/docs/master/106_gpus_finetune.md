# 106. How many GPUs to FINE-TUNE: the maths

## 106.1 The memory equation (chapter 36, applied)

$$ M_{train} ≈ 1.1 × (M_{weights} + 16·N_{trainable} + M_{activations} + M_{logits}) + 1.5 GB

$$ GPUs = ceil( M_{train} / (VRAM × 0.9) )     (only if the model is SHARDED: FSDP / ZeRO-3)

With plain data parallel (DDP), every GPU holds a full copy: more GPUs make training **faster**, not **fit**.

## 106.2 Worked example: 8B model, QLoRA, seq 4096, batch 1 (chapter 36 numbers)

|Term|Value|
|---|---|
|weights (NF4 + fp32 non-quantized parts)|~7.3 GB|
|adapter state (41.9M trainable × 10 bytes, 8-bit Adam)|~0.4 GB|
|activations (checkpointed)|~0.8 GB|
|logits|~1.0 GB|
|total|~11.9 GB → **1 × 24 GB GPU**|

The same model with bf16 LoRA: ~20.6 GB → a 24 GB card is tight; a 48 GB card is comfortable.

## 106.3 How long, and therefore how many GPUs to buy time with

$$ FLOPs ≈ 6 × N_{active} × tokens     (frozen base, gradient checkpointing)
$$ GPU-hours = tokens / (measured tokens/sec per GPU) / 3600
$$ wall-clock = GPU-hours / GPUs       (DDP scales close to linearly while the data loader keeps up)

Measure tokens/sec in the 20-step dry run (runbook step 27) and put it in `--train-tps` of `peftkit.capacity` or `--tokens-per-sec` of `peftkit.plan`.

## 106.4 The command

```any
python -m peftkit.plan <model> --seq-len 4096 --batch 1 --dataset-tokens 2e7 --epochs 2 \
    --tokens-per-sec <measured> --price-per-gpu-hour <your price>
python -m peftkit.capacity --model <model> --users 1 --dataset-tokens 2e7 --train-tps <measured> \
    --prices configs/providers.yaml          # the train_gpus column, per provider GPU
```

:::warn For non-LLM architectures
Chapter 36's activation formula is for transformer LMs. Vision and audio models scale activations with resolution / audio length instead. Use the dry run's measured peak memory; do not trust the formula for them.
:::
