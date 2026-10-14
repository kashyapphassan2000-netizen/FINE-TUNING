# 10. Step 7 — Install the right stack for THIS GPU

:::why The most common first-day failure
A driver too old for your torch build, or a torch build compiled for the wrong CUDA. Everything else waits on this.
:::

@file src/peftkit/gpu_stack.py

```gpu
bash scripts/gpu_preflight.sh          # driver vs CUDA requirement: stops early with a clear message
python -m peftkit.gpu_stack            # detect this GPU and print the exact install plan
python -m peftkit.gpu_stack --gpu H100 # or plan before renting
bash scripts/setup_gpu_vm.sh           # creates BOTH environments
```

@lab lab64_flow_tools

## 10.1 The order, and why each step exists

1. **driver + CUDA** — `nvidia-smi` must show a driver supporting your torch wheel's CUDA.
2. **python env** — `uv venv .venv-train --python 3.11`.
3. **torch matching that CUDA** — the wheel index decides, not pip's default.
4. **core libraries** — transformers, peft, trl, accelerate, datasets.
5. **bitsandbytes** — only for QLoRA (it quantizes `nn.Linear`; not convs, not fused MoE experts).
6. **flash-attn** — optional, slow to build; skip if SDPA already selects a fused kernel.
7. **a SEPARATE serving env** — vLLM pins an older torch. Sharing one env breaks both.
8. **verify** — `torch.cuda.is_available()` is True and the device name is right.

## 10.2 What changes per GPU (from the lab's table)

|GPU|bf16|fp8|FlashAttention-2|Consequence|
|---|---|---|---|---|
|T4, V100|no|no|no|fp16 + loss scaling, `attn_implementation="sdpa"`, expect instability|
|A10G, A100|yes|no|yes|bf16 everywhere; no fp8 KV cache|
|L4, L40S, RTX 4090|yes|yes|yes|fp8 serving possible; 24-48 GB decides the model size|
|H100, H200|yes|yes|yes (FA3)|fp8 weights/KV worth testing|
|B200|yes|yes|yes|verify your torch/vLLM build supports the new compute capability|

Multi-GPU adds: NCCL (ships with torch), an `accelerate` config (DDP or FSDP2), and `nvidia-smi topo -m` to confirm NVLink before a multi-node run.

:::check
- `torch.cuda.is_available()` → True, device name correct.
- Two environments exist, and `which python` shows the training one when you train.
:::
