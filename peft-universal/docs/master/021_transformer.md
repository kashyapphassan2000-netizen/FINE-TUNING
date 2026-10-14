# 21. Transformer internals you must know (and no more)

:::why Why a fine-tuner needs this
`target_modules: ["q_proj","v_proj"]` is meaningless until you know what Q and V are. KV cache sizing, GQA, RoPE and RMSNorm all decide memory and which layers you adapt.
:::

## 21.1 The path of one token

```text One decoder layer, repeated N times
token ids -> embedding lookup -> [ layer ] x N -> final RMSNorm -> LM head -> logits over the vocabulary

inside one layer:
  x -> RMSNorm -> attention(Q,K,V with RoPE) -> + x (residual)
    -> RMSNorm -> MLP (SwiGLU)              -> + x (residual)
```

- **Embedding**: turns each token id into a vector.
- **Attention**: every token looks at earlier tokens and pulls in what it needs. Q ("what am I looking for"), K ("what do I offer"), V ("what I actually pass on"). score = Q·K/√d, softmaxed, used to weight V.
- **Residual (+ x)**: the layer adds a correction instead of replacing the signal, which is why deep stacks train at all.
- **RMSNorm**: keeps activations at a stable scale.
- **MLP (SwiGLU)**: per-token processing; this is where most parameters live.
- **LM head**: projects the final vector to one score per vocabulary token.

**Which parts LoRA adapts**: `q_proj, k_proj, v_proj, o_proj` (attention) and `gate_proj, up_proj, down_proj` (MLP). That is the whole meaning of `all-linear`.

## 21.2 Verified from scratch

Each piece below is implemented in ~5 lines and compared with the real transformers implementation.

@file labs/lab12_transformer_internals.py
@lab lab12_transformer_internals

What each check buys you:

|Piece|Formula|Why a fine-tuner cares|
|---|---|---|
|RMSNorm|x · weight / sqrt(mean(x²) + ε)|1-D tensors; QLoRA's prep upcasts these to fp32|
|RoPE|rotate Q and K by an angle proportional to position|position handling; context-length extension changes its base frequency|
|GQA|H query heads share KV heads|KV cache is H/KV times smaller: the serving-memory lever (chapter 36.5)|
|SwiGLU|down(silu(gate(x)) · up(x))|3 matrices per layer, the biggest LoRA targets|
|KV cache|reuse past K and V|why generation is fast, and why it eats VRAM per user|

:::tip Enough is enough
You do not need to derive attention gradients. You need to know which matrices exist, what their names are, and what each costs in memory.
:::

:::check
- I can draw the layer from memory.
- I can say what Q, K and V do in one sentence each.
- I ran lab 12 and all five checks printed True.
:::
