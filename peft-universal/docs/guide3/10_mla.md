# 10. Multi-Head Latent Attention (MLA)

:::why In one line
Instead of caching K and V per head, cache **one low-rank latent vector per token** and rebuild the per-head K and V from it when you need them. DeepSeek-V2/V3 and Kimi-K2 use it; it is why a 1T model can serve long contexts.
:::

## 10.1 Intuition

GQA saves memory by making heads share. MLA saves memory by **compressing**: the K/V information for all heads is squeezed through a bottleneck of size `kv_lora_rank`, and the per-head vectors are reconstructed by a matrix multiply at use time. You trade a little compute for a lot of memory.

## 10.2 The maths

$$ c_{t} = x_{t} W^{DKV}            (compressed latent, size r = kv_lora_rank)
$$ K_{h} = c_{t} W^{UK}_{h} ,  V_{h} = c_{t} W^{UV}_{h}    (rebuilt per head)

Only `c_t` (plus a small decoupled RoPE key) is cached:

$$ bytes per token = L × (kv_lora_rank + qk_rope_head_dim) × bytes

DeepSeek-V3 (read from its config): `kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `v_head_dim=128`, 128 heads, 61 layers → `61 × 576 × 2 = 68.6 KB per token`, verified by `peftkit.kv` in chapter 3.

**Why a separate RoPE part.** At decode you would like to "absorb" `W^{UK}` into the query projection so the per-head K never has to be materialised. RoPE applies a position-dependent rotation to K, and that rotation does not commute with the absorbed matrix product. The fix in the paper is *decoupled RoPE*: a small extra key dimension (64 here) carries the rotary part and is cached alongside the latent, while the rest of the key stays rotation-free.

## 10.3 From scratch

Lab 34 implements MLA end to end: a down-projection to the latent (which is what gets cached), an up-projection to per-head K and V, and standard attention on top. Measured: 288 KV bytes per token versus 4,096 for MHA at the same width.

## 10.4 Trade-offs, honestly

|Gain|Cost|
|---|---|
|the smallest cache per layer at high head counts|extra matmuls at decode (compute for memory)|
|works with very many heads (128) without cache blow-up|kernels are more complex; fewer implementations support it|
|enables long context on trillion-parameter models|LoRA target names differ (`kv_a_proj_with_mqa`, `kv_b_proj`, `q_a_proj`, `q_b_proj`)|
|vLLM supports it, including an fp8 MLA cache dtype (`fp8_ds_mla` appears in its cache config)|adapters and tooling written for `q_proj/k_proj/v_proj` need changing|

## 10.5 Common mistakes

- Sizing an MLA model's cache with the GQA formula. It overestimates by an order of magnitude.
- Copying `target_modules: ["q_proj","k_proj","v_proj"]` onto a DeepSeek-style model: those names do not exist. Run `peftkit.compat` and read the printed names.
- Assuming MLA is always smallest: at low head counts a 1-KV-head MQA model can cache less (lab 34: MQA 256 vs MLA 288 bytes/token at 16 heads). MLA wins as head count grows.

:::check
- I can write MLA's per-token cache formula and evaluate it for DeepSeek-V3.
- I know why the decoupled RoPE dimension exists.
:::
