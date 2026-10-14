# 12. Positional encoding schemes

:::why In one line
Attention is order-blind: shuffle the tokens and the maths gives the same answer. Positional encoding is how order gets in, and the scheme you choose decides how far beyond the training length the model can still work.
:::

## 12.1 The problem

`softmax(QK^T/√d)V` is permutation-equivariant. Causal masking tells the model which tokens are *before* it, but not *how far* before. Something must encode distance.

## 12.2 The families

@file labs/lab37_positional_schemes.py
@lab lab37_positional_schemes

|Family|Where it acts|Parameters|Beyond training length|
|---|---|---|---|
|sinusoidal absolute|added to the embedding|none|defined, but quality degrades|
|learned absolute|added to the embedding|`max_pos × d`|impossible past `max_pos`|
|relative bias (T5-style)|added to scores|small table|moderate|
|ALiBi|linear penalty on scores|none|good by construction|
|RoPE|rotates Q and K|none|needs scaling (chapter 13)|
|NoPE|nothing|none|architecture dependent|

## 12.3 Sliding windows: a positional decision with a memory effect

Modern models mix layer types. Gemma-3's default config (read here) alternates **five sliding-attention layers to one full-attention layer**, with two different RoPE bases: theta 10,000 on local layers, 1,000,000 on global ones. gpt-oss alternates sliding (window 128) and full layers, with YaRN scaling (factor 32, original length 4,096).

Why this matters here: a sliding layer's KV cache **stops growing** once the window is full. A model with 5:1 sliding:full layers caches roughly `(5·window + 1·full)/6` per token instead of full length — which is why such models advertise long contexts at modest memory. `peftkit.kv` reports the split; the plain formula in chapter 3 over-estimates for them.

## 12.4 Common mistakes

- Serving a model at a longer context than its config's `max_position_embeddings` without any scaling, then blaming the model for degraded output.
- Changing rope settings on a server that has a warm prefix cache (chapter 11) — cached keys carry their positions.
- Assuming ALiBi and RoPE are interchangeable: ALiBi biases scores and caches raw K; RoPE rotates K before caching.
- Treating perplexity as proof of long-context ability. Use retrieval-style tests (find a fact planted deep in the context).

:::check
- I know which scheme my model uses and what its trained context length is.
- I know whether any of its layers are sliding-window.
:::
