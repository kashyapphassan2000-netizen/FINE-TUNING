# 5. PyTorch SDPA: the unified attention API

:::why In one line
One function, several kernels. You write the same call; PyTorch picks FlashAttention, a memory-efficient kernel, cuDNN, or the plain maths path depending on your hardware, dtype and mask.
:::

## 5.1 The API (read from torch 2.14.0)

```text signature
scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                             is_causal=False, scale=None, enable_gqa=False) -> Tensor
```

Backends available in this build: `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`, `MATH`, `OVERRIDEABLE`. The first three are CUDA-only; `MATH` is the portable reference implementation and is what runs on CPU.

## 5.2 Verified behaviour

@file labs/lab33_sdpa.py
@lab lab33_sdpa

Four things that lab proves, and that people get wrong:

1. `enable_gqa=True` does the KV-head repetition for you — no manual `repeat_interleave`.
2. `is_causal=True` equals passing an explicit upper-triangular `-inf` mask, but is cheaper: the kernel skips masked blocks instead of computing and then discarding them.
3. `scale` overrides the default `1/sqrt(head_dim)`.
4. Forcing the `MATH` backend gives the same numbers as the default choice.

## 5.3 Choosing a backend deliberately

```py force a backend (useful for debugging a suspected kernel issue)
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.MATH):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

In transformers 5.17 the equivalent switch is `attn_implementation`, and the registry contains: `sdpa`, `eager`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`, `flex_attention`, plus `paged|…` variants used by its paged-attention integration.

|Implementation|Use it when|
|---|---|
|`sdpa`|default; good everywhere|
|`eager`|debugging: it materialises the scores so you can inspect them|
|`flash_attention_2/3/4`|long sequences on supported GPUs, package installed|
|`flex_attention`|custom masks/biases (sliding windows, document masks) without writing kernels|

## 5.4 Common mistakes

- Passing `is_causal=True` on a **decode** step (one query, many cached keys). That masks the past. Use `False`.
- Passing both `is_causal=True` and an `attn_mask`; semantics get confusing fast — pick one.
- Float masks vs boolean masks: a boolean mask means "keep where True"; a float mask is **added** to the scores, so it must contain `-inf`, not `0/1`.
- Expecting a fused kernel with fp32 inputs on CUDA: most fused paths need fp16/bf16.
- Benchmarking with `MATH` forced and concluding attention is slow.

:::check
- I know what `enable_gqa` does and when `is_causal` must be False.
- I can force a backend to test whether a kernel is the problem.
:::
