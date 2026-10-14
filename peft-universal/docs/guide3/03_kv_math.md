# 3. KV cache memory math

:::why The formula to memorise
$$ bytes per token = 2 × layers × kv_heads × head_dim × bytes_per_element
Multiply by tokens in flight (sum of all sequence lengths being served). That is your KV cache.
:::

## 3.1 Verified against reality

@file labs/lab31_kv_cache_math.py
@lab lab31_kv_cache_math

The formula predicted 4,096 bytes and PyTorch allocated exactly 4,096. The 2 is K and V; there is nothing else in there.

## 3.2 Reading the table

For Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, bf16): **128 KB per token**. That number does the work:

- one 128k-token conversation = **16 GB** of KV — more than the model's own weights on many cards;
- 256 users at 8k tokens each = **256 GB**, which no single GPU has;
- switching KV to fp8 halves both numbers;
- the MLP size does not appear anywhere, which is why a huge MoE can have a small cache.

MLA changes the shape of the formula entirely: DeepSeek-V3 caches `layers × (kv_lora_rank + qk_rope_head_dim)` = 61 × 576 = **68.6 KB per token** with 128 attention heads. A GQA model of the same depth with 8 KV heads costs 244 KB per token.

## 3.3 The calculator

@file src/peftkit/kv.py

```any Any model: Hub id, local folder, or config.json
python -m peftkit.kv meta-llama/Llama-3.1-8B-Instruct --context 32768 --gpu-gb 80 --gpus 1 --kv-dtype bf16
python -m peftkit.kv configs/models/deepseek-v3.config.json --context 131072 --gpu-gb 141 --gpus 8 --weight-gb 700
```

```out REAL OUTPUT (DeepSeek-V3 config, 8 x 141 GB, weights given as 700 GB)
DeepSeek-V3  attention=mla layers=61 heads=128 kv_heads=128 head_dim=64
KV per token (bf16): 68.625 KB   one 131072-token sequence: 8.578 GB
weights 700.0 GB on 8x141.0 GB -> KV budget 426.5 GB
fits ~6,516,832 cached tokens = 49 sequences at 131072 tokens
```

:::warn Read the caveat
For MLA models the "MHA / GQA / MQA" comparison rows in that tool use the config's `head_dim`, while DeepSeek's real per-head K dimension is 192 (128 no-rope + 64 rope) and V is 128. The MLA row is exact; the comparison rows are indicative.
:::

## 3.4 Capacity planning, the way a serving team does it

$$ KV budget = GPUs × VRAM × utilisation − weights − activation/workspace overhead
$$ concurrent sequences ≈ KV budget / (bytes per token × average tokens per sequence)

vLLM does this at start-up and prints its own "maximum concurrency" figure; trust that over any estimate, including this book's.

Levers, in the order you should reach for them:

|Lever|Effect|Cost|
|---|---|---|
|shorter `max_model_len`|linear|users cannot send long prompts|
|fp8 KV cache|halves KV|small accuracy risk; needs kernel support|
|a model with fewer KV heads (GQA/MQA) or MLA|2-30x|it is an architecture choice, not a flag|
|prefix caching|removes duplicate system prompts|only helps shared prefixes|
|more GPUs (tensor parallel)|linear in KV budget too|money, and TP must divide the head count|
|sliding-window models|bounded per layer|architecture choice|

## 3.5 Trillion-scale, honestly

At 1T parameters the weights dominate memory (Volume 1, chapter 34), but the cache still decides how many users you serve. This is exactly why the two trillion-scale open models (DeepSeek-V3, Kimi-K2) both use MLA plus MoE: MoE keeps compute per token low, MLA keeps the cache small, and the weights are spread over many GPUs. If a future 10T model keeps MHA, its serving cost per user will be dominated by the cache, not the weights.

:::check
- I can compute bytes/token for my model without looking it up.
- I know my KV budget after weights on my chosen GPU.
- I checked whether my model has sliding-window or MLA before trusting the plain formula.
:::
