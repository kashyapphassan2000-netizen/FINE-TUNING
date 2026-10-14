# 9. Grouped-Query Attention (GQA)

:::why In one line
Split the query heads into G groups; each group shares one K/V head. G = H is MHA, G = 1 is MQA, and everything interesting is in between. This is what almost every modern open model uses.
:::

## 9.1 The maths

With `H` query heads and `G` KV heads (`H mod G = 0`), query head h uses KV head `⌊h/(H/G)⌋`:

$$ head_{h} = softmax(Q_{h} K_{⌊h/(H/G)⌋}^{T} / sqrt(d_h)) V_{⌊h/(H/G)⌋}

Cache per token: `2 × L × G × d_h × bytes`, an `H/G` reduction versus MHA.

## 9.2 Real configurations

|Model|heads|KV heads|ratio|KV per token (bf16)|
|---|---|---|---|---|
|Llama-3.1-8B|32|8|4:1|128 KB|
|Llama-3.1-70B|64|8|8:1|327 KB|
|Llama-3.1-405B|128|8|16:1|516 KB|
|Qwen3-235B-A22B|64|4|16:1|192 KB|

Notice the pattern: as models get wider, the KV-head count stays at 4-8. The industry converged on "a handful of KV heads is enough", which is the practical claim of the GQA paper (Ainslie et al., 2023) — near-MHA quality at close to MQA cost.

## 9.3 From scratch

Lab 34 implements all of it: the KV projection is `G × d_h` wide instead of `H × d_h`, and the heads are repeated before the attention call (`repeat_interleave`, or `enable_gqa=True` in SDPA, which is what production kernels do internally without materialising the copy).

## 9.4 Tensor parallelism interacts with G

When serving with tensor parallel size `TP`, KV heads are split across GPUs. If `TP > G` the KV heads must be replicated (vLLM handles this), which means the KV cache is duplicated per GPU and your effective capacity stops improving. **Rule: keep `TP ≤ kv_heads` when you can**; with 8 KV heads, TP 8 is the natural maximum before replication.

## 9.5 Common mistakes

- Choosing `TP=16` on a model with 8 KV heads and wondering why KV memory did not halve.
- Reading `num_key_value_heads` as "attention heads" when sizing the cache. The KV head count is the one in the formula.
- Assuming GQA changes the parameter count much: it mostly shrinks `W^K` and `W^V` (lab 34: 4.19M → 2.62M attention parameters at 16→4 KV heads).

:::check
- I know my model's H:G ratio and its KV bytes per token.
- My planned tensor-parallel size does not exceed the KV head count.
:::
