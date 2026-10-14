# 6. Attention variants: one interface, four memory profiles

Every variant answers the same question — *how many distinct K/V vectors do we store per token?* — and they all keep the same outside interface, so a serving stack can support all of them.

@file labs/lab34_attention_variants.py
@lab lab34_attention_variants

The four chapters that follow are the detail. The measured table above is the summary: for the same model width (1024, 16 query heads), KV per token is **4,096 / 1,024 / 256 / 288 bytes** for MHA / GQA(4) / MQA / MLA(rank 128).

## 6.1 How to spot which one a model uses

|Look at `config.json`|Meaning|
|---|---|
|`num_key_value_heads` == `num_attention_heads`|MHA|
|`num_key_value_heads` == 1|MQA|
|`1 < num_key_value_heads < num_attention_heads`|GQA|
|`kv_lora_rank` present|MLA|
|`layer_types` with `sliding_attention`|part of the cache is window-bounded (chapter 12)|

```any
python -m peftkit.kv <model>        # prints the variant, the per-token cost, and what the alternatives would cost
python -m peftkit.compat <model>    # prints the layer names you would adapt with LoRA
```

## 6.2 The trade in one sentence each

- **MHA**: maximum expressiveness per layer, maximum cache.
- **MQA**: minimum cache, some quality loss reported in the original work.
- **GQA**: the compromise that won; near-MHA quality, MQA-like cache.
- **MLA**: compress K/V into a latent vector, rebuild per head on the fly; smallest cache at high head counts, most complex kernels.

:::warn You cannot change this after pretraining
Converting MHA → GQA ("uptraining") requires extra training and is a research-grade operation, not a config edit. For a fine-tuning or serving project, treat the variant as fixed and choose your base model accordingly.
:::
