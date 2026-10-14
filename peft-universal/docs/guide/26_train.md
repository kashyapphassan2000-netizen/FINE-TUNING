# 26. Training

@file src/peftkit/train.py

What `train.py` does, in order: loads and validates config → loads jsonl (keeps only `messages`) → tokenizer (pad and template fallbacks) → model (4-bit if requested, gradient checkpointing, input grads) → computes total optimizer steps (needed by AdaLoRA) → wraps with PEFT → LoRA+ optimizer or AdaLoRA callback if needed → `SFTTrainer` → saves `adapter/` and `run_meta.json` (config, versions, GPU, trainable params, loss history, peak VRAM, AdaLoRA ranks, eval) → optional push.

:::warn API facts checked in transformers 5.17 / TRL 1.13
- `warmup_ratio` no longer exists; `warmup_steps` accepts a float ratio.
- `SFTConfig` uses `max_length` (not `max_seq_length`) and defaults `bf16` to on, so CPU runs need `bf16=False, use_cpu=True` (the template sets these automatically).
- `from_pretrained(..., dtype=...)` (the old name was `torch_dtype`).
- TRL default loss `chunked_nll` lowers logits memory.
:::

## 26.1 Laptop smoke run (CPU, tiny model)

`pytest` already trains all four CPU-capable methods. That is the smoke test.

## 26.2 GPU: 20-step dry run, then the real run

```gpu Dry run (inside tmux)
tmux new -s train
source .venv-train/bin/activate
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/llama-3.1-8b.yaml \
   --set train.max_steps=20 train.logging_steps=5 train.output_dir=outputs/dryrun
python -c "import json;m=json.load(open('outputs/dryrun/run_meta.json'));print(m['peak_vram_gb'],m['train_seconds'])"
```

Tokens/second estimate from the dry run: `train_seconds / 20` is seconds per optimizer step; each step processes `per_device × grad_accum × GPUs × avg_tokens`. Multiply to estimate the full run and cost.

```gpu Full run (any method: change the second file)
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/llama-3.1-8b.yaml
# detach: Ctrl-b d      re-attach: tmux attach -t train      resume after a crash:
python -m peftkit.train --config ... --resume true
```

```out Last line shape
{"method": "qlora", "trainable_params": 41943040, "global_step": <N>, "train_loss": <x>}
```

## 26.3 Multi-GPU

@file configs/accelerate/ddp.yaml
@file configs/accelerate/fsdp2.yaml

```gpu
# DDP: same memory per GPU, more throughput (works with QLoRA)
accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 4 -m peftkit.train \
   --config configs/base.yaml configs/methods/qlora.yaml configs/models/llama-3.1-8b.yaml
# FSDP2: shard a bf16 base that does not fit on one GPU (LoRA/DoRA/LoRA+). Not executed in this guide.
accelerate launch --config_file configs/accelerate/fsdp2.yaml --num_processes 4 -m peftkit.train \
   --config configs/base.yaml configs/methods/lora.yaml configs/models/qwen-large.yaml \
   --set train.gradient_checkpointing=false
```

:::warn
`accelerate launch -m module` support and FSDP2 + PEFT behaviour change between versions. If the launch fails, run `accelerate launch ... src/peftkit/train.py ...` instead, and check the accelerate/PEFT FSDP docs for your versions. PEFT warns when an adapter is saved from un-gathered FSDP shards; check that `adapter_model.safetensors` has non-empty tensors.
:::

## 26.4 All-in-one

@file scripts/pipeline.sh

```gpu
METHOD=dora MODEL_CFG=configs/models/llama-3.1-8b.yaml SKIP_SYNTH=1 bash scripts/pipeline.sh
```

## 26.5 Reading the training log

|Signal|Meaning|Action|
|---|---|---|
|loss flat from step 0|adapter not learning|check trainable params > 0, lr not 2e-5|
|loss → ~0 quickly, eval rises|overfitting|fewer epochs, lower r, more data|
|loss NaN|fp16 overflow or lr too high|bf16, lower lr, `max_grad_norm: 1.0`|
|eval_mean_token_accuracy rises|learning the answers|good|
|AdaLoRA loss higher than LoRA|includes orthogonal penalty|compare final `run_meta.json` eval|

:::check
- Dry run finished; `peak_vram_gb` recorded; batch size chosen.
- Full run finished; `outputs/<run>/adapter/` and `run_meta.json` exist.
:::
