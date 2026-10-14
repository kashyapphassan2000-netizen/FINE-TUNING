# 4. FlashAttention

:::why In one line
Attention does not need the full N×N score matrix in memory. Stream over blocks, keep a running max and sum, rescale as you go. The result is exact, the memory is linear, and on a GPU it is much faster because it stops moving that matrix to and from HBM.
:::

## 4.1 Intuition

Averaging a million numbers does not require writing them all down: keep a running total and a count. Softmax is harder because of the exponentials and the max-subtraction for stability, but the same trick works — if a later block contains a bigger maximum, you rescale what you have so far by `exp(old_max − new_max)`. Nothing is approximated.

## 4.2 The problem it solves

Standard attention stores the scores: `N × N` per head per layer. At 32k context that is 2 GB **per head** in bf16. Even when it fits, attention is **memory-bandwidth bound** — the GPU spends its time writing and re-reading that matrix from HBM rather than doing maths.

## 4.3 The maths

For blocks of keys, maintaining running max `m`, running denominator `l` and running output `o`:

$$ m_{new} = max(m, max(s_{block}))
$$ p = exp(s_{block} − m_{new}) ,   correction = exp(m − m_{new})
$$ l ← l · correction + Σ p
$$ o ← o · correction + p · V_{block}
$$ output = o / l

## 4.4 From scratch, and exact

@file labs/lab32_flash_attention.py
@lab lab32_flash_attention

Max difference from standard attention: **2.38e-07**, i.e. float noise. This is the whole idea behind FlashAttention; the real kernels add GPU specifics: tiles sized for SRAM, fused softmax, recomputation in the backward pass instead of storing the matrix, and (v2/v3) better work partitioning and Hopper-specific asynchronous copies.

## 4.5 Using it, rather than writing it

|Route|How|Notes|
|---|---|---|
|torch SDPA|`F.scaled_dot_product_attention(...)`|picks a fused kernel when it can (chapter 5)|
|transformers|`attn_implementation="flash_attention_2"` (also `_3`, `_4` in 5.17)|needs the `flash-attn` package (2.8.3.post1 on PyPI, 18 Sep 2026) and a supported GPU|
|vLLM|its own attention backends|chosen automatically per hardware|

:::warn Honest limits
- FlashAttention is CUDA/ROCm kernel work: **none of it runs on CPU**, so the timing claims in this chapter are structural, not benchmarked here.
- Building `flash-attn` from source is slow and version-sensitive (torch/CUDA must match). If `sdpa` already selects a fused kernel, you often gain nothing by installing it.
- It changes speed and memory, not results (beyond float noise). If your loss changes when you switch backends, something else is wrong — usually masking.
- FlashAttention needs the right dtype (fp16/bf16) and head-dim limits; with an unsupported combination it silently falls back to another backend.
:::

:::check
- I can explain why the N×N matrix is the memory problem.
- I ran lab 32 and saw the streaming result match to ~1e-7.
- I know which attn_implementation my training run actually used.
:::
