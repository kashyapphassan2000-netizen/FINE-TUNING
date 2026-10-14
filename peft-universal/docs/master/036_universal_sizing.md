# 36. Universal sizing for any model, from millions to trillions of parameters

:::honest What this addendum adds and how it was verified
- New code: `src/peftkit/archs.py` (architecture analysis for any causal LM), `src/peftkit/plan.py` (memory, GPU count, multi-node, KV cache, compute, time, cost), `src/peftkit/adapters.py` (multi-adapter registry), a QLoRA option for MoE models, single-process multi-GPU loading, per-key adapter permissions in the gateway, 6 reference configs.
- Test suite after the addendum: **47 passed, 1 skipped** (the skip is the GPU-only QLoRA test). Chapter 0 still shows the earlier count of 32; this line supersedes it.
- The method was validated by rebuilding models on PyTorch's `meta` device (no memory, no weights) with transformers 5.17 and counting: DeepSeek-V3 **671.0B total / 37.6B active**, Llama-3.1-405B **405.9B**, Qwen3-235B-A22B **235.1B / 22.2B**, Mixtral-8x7B **46.7B / 12.9B**, Llama-3.1-8B **8.03B**, and a Kimi-K2 configuration rebuilt from Moonshot's published architecture table **1026.4B / 32.9B**. These match the sizes the vendors publish.
- Everything about GPU memory is still an estimate. Prices are never included: you supply them.
:::

## 36.1 The two questions you are really asking

1. **Does it fit?** (memory: VRAM for weights, training state, activations, and at serving time the KV cache)
2. **How long and how much?** (compute: FLOPs → GPU-hours → money)

Memory decides the *minimum* number of GPUs. Compute decides how many *more* GPUs you add to finish faster. Always answer them in that order.

## 36.2 VRAM vs KV cache: the difference in one table

|Term|What it is|Grows with|Matters for|
|---|---|---|---|
|VRAM|the GPU's memory. Everything below must fit inside it|-|always|
|weights|the model's parameters|parameter count × bytes per parameter|training and serving|
|optimizer + gradients|state for the trainable parameters|trainable params (tiny with PEFT)|training|
|activations|intermediate results kept for backward|batch × sequence × hidden × layers|training|
|logits|scores over the vocabulary per token|batch × sequence × vocabulary|training (and long prefills)|
|**KV cache**|keys/values of past tokens so generation does not recompute them|tokens in flight × layers × KV size|**serving**|

So: for **training**, KV cache is irrelevant (the template sets `use_cache=False`). For **serving**, KV cache is usually what decides how many users you can serve at once, once the weights fit.

## 36.3 The universal memory equations

:::math Weights
$$ M_{weights} = N_{total} × b
b = bytes per parameter: fp32 4, bf16 2, fp8 1, int8 1, MXFP4 ≈ 0.53, NF4 + double quant ≈ 0.52.
**MoE: use N_{total}, not N_{active}.** Every expert must be in memory even though each token uses only a few.
:::

:::math PEFT training (per replica)
$$ M_{train} ≈ 1.1 × (M_{weights} + 16 × N_{trainable} + M_{act} + M_{logits}) + 1.5 GB
- 16 bytes per trainable parameter = fp32 weight 4 + gradient 4 + Adam moments 8 (10 with 8-bit Adam).
- M_{act} with gradient checkpointing ≈ layers × 2 × seq × batch × hidden bytes + one full layer's activations.
- M_{logits} ≈ batch × seq × vocab × 4 bytes (TRL's chunked loss reduces it).
:::

:::math Serving
$$ M_{serve} ≈ (M_{weights} + KV_{token} × tokens_{in flight}) / 0.9 + 1.5 GB
KV_{token} depends on the attention type (34.5). tokens in flight ≈ concurrent sequences × their length.
:::

:::math GPU count
$$ GPUs = ceil( M / (VRAM_{per GPU} × 0.9) )
More than 1 GPU for training only helps memory if the model is **sharded** (FSDP / ZeRO-3 / expert or pipeline parallel). For serving, GPUs = tensor-parallel (and for very large MoE, expert-parallel) size, usually a power of two.
More than 8 GPUs means **more than one node**: you then need a fast inter-node network (InfiniBand/RoCE class), a multi-node launcher, and a framework built for it.
:::

## 36.4 Dense vs MoE: total vs active

|Quantity|Dense model|MoE model|
|---|---|---|
|memory for weights|N|N_{total} (all experts)|
|compute per token|≈ 2N (forward)|≈ 2N_{active}|
|LoRA on `all-linear`|MLPs + attention|**every expert** gets adapters: can be billions of trainable params|
|typical PEFT choice|all-linear|attention (+ shared expert / dense layers), experts frozen|

Measured with the planner on the Kimi-K2-like configuration at r = 16: LoRA on all linear layers including all 384 experts per layer = **11,770,385,408** trainable parameters (~175 GB of optimizer state). Attention-only = **61,027,328**. That is a 190x difference from one YAML line.

```yaml Attention-only targets for an MoE model (names come from `peftkit.plan` / `inspect_model` output)
method:
  target_modules: [q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj]   # DeepSeek/Kimi MLA names
```

## 36.5 KV cache per attention type

|Attention type|Seen in|KV bytes per token (all layers)|
|---|---|---|
|MHA|older models|2 × L × heads × head_dim × b|
|GQA / MQA|Llama 3, Qwen, Mistral|2 × L × kv_heads × head_dim × b|
|MLA (multi-head latent)|DeepSeek-V3, Kimi-K2|L × (kv_lora_rank + qk_rope_head_dim) × b|
|sliding-window mix|Gemma 2/3, gpt-oss|full layers as GQA; sliding layers stop growing after the window|
|hybrid linear attention / SSM|Qwen3-Next, Falcon-H1, Jamba, Kimi-Linear|only the full-attention layers grow; the rest keep a constant-size state|

Worked numbers (bf16, from the tests): Llama-3.1-8B = 2 × 32 × 8 × 128 × 2 = **131,072 bytes = 0.125 MB per token**. DeepSeek-V3 = 61 × (512 + 64) × 2 = **70,272 bytes ≈ 0.067 MB per token** for a model 84x larger. MLA is why these giant MoE models are servable at long context. FP8 KV cache (where the server supports it) halves these numbers.

:::math Worked example: how many users?
Llama-3.1-8B, bf16, one H100 80 GB, gpu-memory-utilization 0.9:
free for KV ≈ 80 × 0.9 − 15 (weights) − 1.5 ≈ 55.5 GB ≈ 444,000 tokens at 0.125 MB/token.
At 8,192 tokens per conversation that is about 54 full-length conversations at once. At 2,048 tokens, about 216.
vLLM prints its own number ("maximum concurrency") at start-up. Trust that over this estimate.
:::

## 36.6 The planner

@file src/peftkit/archs.py
@file src/peftkit/plan.py

```any Any model: Hub id (downloads config.json only), local folder, or config file
python -m peftkit.plan meta-llama/Llama-3.1-8B-Instruct --seq-len 4096 --serve-len 32768 --concurrency 32
python -m peftkit.plan deepseek-ai/DeepSeek-V3 --targets attention --serve-dtype fp8
python -m peftkit.plan <org>/<new-model> --trust-remote-code        # only for repos whose code you have read
python -m peftkit.plan configs/models/kimi-k2-like.config.json --dataset-tokens 5e7 --epochs 2 \
    --tokens-per-sec <measured> --price-per-gpu-hour <your price>
```

How it stays universal:

1. `config.json` is read and normalised: 9 families of field names (e.g. `n_routed_experts` / `num_local_experts` / `num_experts`) map to one spec; multimodal repos' nested `text_config` is unwrapped.
2. The model is built on the `meta` device with the **real** transformers code and counted exactly (total, experts, embeddings, linear vs fused tensors), and PEFT is applied on meta to count trainable parameters exactly for all-linear and attention-only targets.
3. If the architecture cannot be built locally (custom code, missing dependency), it falls back to the dense formula of chapter 35 and says so in the output (`source`).
4. Attention type is detected from the config (MLA fields, `layer_types`, KV heads) for the KV formula.
5. Pre-quantized checkpoints (`quantization_config`) are flagged.

## 36.7 The size ladder (real planner output)

@lab plan_size_ladder

How to read it: "LoRA" and "QLoRA" are GiB for one model replica with all-linear targets; "LoRA-attn" is attention-only; "train H100-80" is the GPU count if the model is sharded; "serve" is total serving memory for the stated load and the GPU counts per type (powers of two).

:::warn Brutal truths the ladder shows
- **Up to ~13B dense:** one GPU. 24 GB cards work with QLoRA; 40-48 GB with LoRA.
- **30-70B dense:** QLoRA on one 80 GB GPU, or LoRA sharded over 2-4 GPUs.
- **100B-400B:** a full 8-GPU node, FSDP sharding, and serious engineering. Expect restarts; checkpoint often.
- **Trillion-parameter MoE:** at least 2 nodes of the largest GPUs for bf16 LoRA. This template's single-node HF stack is not the right tool there: use frameworks built for expert/pipeline parallel training (examples of such stacks: DeepSpeed, Megatron-LM / NVIDIA NeMo, torchtitan, ms-swift, LLaMA-Factory; check each one's current support for your model) or use a provider's managed fine-tuning if one exists for that model.
- **QLoRA does not help fused-expert MoE models in this stack** (34.8).
:::

## 36.8 Two MoE traps verified in the source

:::warn Trap 1: bitsandbytes does not quantize fused experts
In transformers 5.17, MoE experts are stored as 3-D `nn.Parameter` tensors (e.g. `gate_up_proj` of shape experts × 2·inter × hidden), not `nn.Linear`. `transformers/integrations/bitsandbytes.py` replaces only `nn.Linear` (and `Conv1D`). So in "QLoRA", the experts, which are 90%+ of the parameters, stay unquantized.
:::

:::warn Trap 2: PEFT's k-bit prep upcasts them to fp32
`prepare_model_for_kbit_training` casts every non-4-bit bf16/fp16 tensor to fp32. Test in this repo: `experts.gate_up_proj` bf16 → float32. For the Kimi-K2-like config the planner shows QLoRA at ~4.3 TB vs ~2.3 TB for plain bf16 LoRA.

Fix provided: `method.quant.upcast_non_quantized: false` uses `light_kbit_prep` (only 1-D norm/bias tensors go to fp32; experts, embeddings and LM head stay bf16; tested). Even then QLoRA ≈ LoRA for such models. Real memory savings for giant MoE need expert quantization supported by your training framework, or FP8 training support. Check before you rent.
:::

:::tip Good news verified in the same stack
PEFT 0.21 + transformers 5.17 *do* attach LoRA to fused expert tensors with `all-linear` (tested on a tiny Qwen3-MoE), and vLLM 0.29 has MoE LoRA support (2-D and 3-D adapter formats; flag `enable_mixed_moe_lora_format` in its LoRA config). Older library versions may silently skip experts: always print trainable parameter names once.
:::

## 36.9 Pre-quantized checkpoints (FP8, MXFP4, GPTQ/AWQ)

Many large releases ship quantized weights (e.g. FP8 for some DeepSeek/Kimi releases, MXFP4 experts for gpt-oss). For **serving**, vLLM can often load them directly (check its supported quantization list for your version). For **PEFT training** with this template, the safe path is a bf16 copy: use the vendor's official conversion script if one is provided, or load with transformers' dequantization support for that format, then save bf16. The planner flags `quantization_config` so you do not miss it. Disk for a bf16 copy of a 1T model: ~2 TB. Budget it.

## 36.10 Everything else a big run needs

|Resource|Rule|Example (Kimi-K2-like, bf16)|
|---|---|---|
|disk|download size + converted copy + merged copy + checkpoints|~2 TB per bf16 copy; plan 6-8 TB of fast NVMe|
|download time|size ÷ network speed|2 TB at 10 Gbit/s ≈ 1,600 s theoretical; real Hub downloads are slower. Pre-download once to shared storage|
|host RAM|loading is memory-mapped with safetensors, but plan ≥ the largest shard × workers + OS|≥ 1 TB per node is common for these nodes|
|inter-GPU link|NVLink/NVSwitch inside a node for tensor/expert parallel|required|
|inter-node network|InfiniBand/RoCE class for multi-node|required above 8 GPUs|
|checkpoint frequency|≤ every 30-60 min of GPU time|failures are normal at this scale|

:::check
- I ran `peftkit.plan` for my model and read the `source` line (exact vs formula).
- For MoE, I chose target modules deliberately (all-linear vs attention).
- I know my serving KV per token and my expected tokens in flight.
- If the plan says multi-node, I have a framework and a network for it, or I chose a smaller model.
:::
