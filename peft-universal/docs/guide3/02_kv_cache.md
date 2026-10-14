# 2. The KV cache

:::why In one line
Store K and V for every token you have already processed, so each new token only computes its own K and V and attends against the stored ones.
:::

## 2.1 What is cached, and what is not

For each layer and each **KV head**: two tensors, K and V, of shape `(batch, kv_heads, tokens, head_dim)`.

- **Q is not cached.** The query is only needed for the current token.
- **The MLP has no cache.** It is per-token and stateless.
- **RoPE is applied to K before it is stored**, so a cached key already carries its position (chapter 13 explains why that matters).

```text one decode step
new token ──> Q,K,V for this token
              │        │
              │        └─> append K,V to the cache  (cache grows by 1 token)
              └─> attend(Q, all cached K, all cached V) ──> output
```

## 2.2 The maths of one decode step

With a cache of t tokens, for one query token in one head:

$$ scores = q · K_{cache}^{T} / sqrt(d)      (1 × t)
$$ out = softmax(scores) · V_{cache}          (1 × d)

No causal mask is needed: the single query is allowed to see every cached key. (Passing `is_causal=True` on a 1-token query is a classic bug — it would mask everything except the last position.)

## 2.3 From scratch, and with the library

@file labs/lab33_sdpa.py

Section 5 of that lab is exactly a decode step: `(1,H,1,D)` query against a `(1,KV,513,D)` cache.

In transformers, the cache is an object (`past_key_values`) that `generate()` manages for you; `model(ids, use_cache=True)` returns it, and passing it back makes the next call incremental. Cache classes exist for variants — static shapes for CUDA graph capture, offloading, sliding-window and quantized caches — and each trades memory for speed or accuracy.

## 2.4 What it costs you

The cache is the thing that grows without bound: linearly in tokens, in users, in layers and in KV heads. Chapter 3 turns that into bytes, and the rest of the book is a catalogue of ways to shrink or manage it:

|Technique|Chapter|What it attacks|
|---|---|---|
|GQA / MQA|8, 9|fewer KV heads|
|MLA|10|cache a compressed latent instead of per-head K/V|
|sliding-window layers|12|stop the cache growing on most layers|
|fp8 KV cache|3|fewer bytes per number|
|PagedAttention + prefix sharing|11|stop wasting the memory you allocated|

## 2.5 Common mistakes

- Leaving `use_cache=True` during training (wasted memory; the trainer sets it False for a reason).
- Reusing a cache built with one rope configuration for a different one.
- Mixing left and right padding: cached positions must line up with the attention mask.
- Assuming the cache is free to copy; for beam search or n>1 sampling, copying it per beam is exactly what paging avoids (chapter 11).

:::check
- I can name what is stored per layer and why Q is not.
- I know why a decode step does not need a causal mask.
:::
