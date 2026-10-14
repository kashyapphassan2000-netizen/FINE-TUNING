# 7. Multi-Head Attention (MHA)

:::why In one line
The original design: every head has its own Q, K and V projections, so every head caches its own keys and values.
:::

## 7.1 The maths

For head h of H, with model width `d` and head dim `d_h = d/H`:

$$ Q_{h} = x W^{Q}_{h} ,  K_{h} = x W^{K}_{h} ,  V_{h} = x W^{V}_{h}
$$ head_{h} = softmax(Q_{h} K_{h}^{T} / sqrt(d_h)) V_{h}
$$ out = concat(head_{1..H}) W^{O}

Cache per token: `2 × L × H × d_h × bytes`. With `H × d_h = d`, that is `2 × L × d × bytes` — **independent of how you split the heads**, and the largest of all the variants.

## 7.2 Worked number

Llama-3.1-70B's shape (80 layers, 64 heads, head_dim 128) **if it used MHA**: `2 × 80 × 64 × 128 × 2 = 2.6 MB per token`. One 32k conversation would need 85 GB of KV. It does not use MHA: it uses GQA with 8 KV heads, giving 327 KB/token. That single design choice is an 8× difference in serving capacity.

## 7.3 Where it still appears

Older and smaller models (GPT-2 family, original Llama-1 7B/13B, many encoders), and some recent models at small scale where the cache is not the bottleneck. Encoder-only models (BERT-style) also use MHA, but they have no autoregressive decode and therefore no KV cache problem at all.

## 7.4 Common mistakes

- Assuming "more heads = more memory". For a fixed model width, MHA's cache depends on width, not head count.
- Thinking heads are independent models: they share the same input and their outputs are mixed by `W^O`.
- Setting `num_key_value_heads = num_attention_heads` in a config "to be safe" — you have just chosen the most expensive cache.

:::check
- I can write the per-token cache formula for MHA and see why it reduces to 2·L·d·bytes.
:::
