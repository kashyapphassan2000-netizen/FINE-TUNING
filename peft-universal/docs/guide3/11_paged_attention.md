# 11. PagedAttention and vLLM

:::why In one line
Stop reserving a contiguous block of KV memory for each sequence's worst case. Store the cache in fixed-size blocks with a per-sequence block table, exactly like virtual memory pages. Waste collapses, and sequences can share blocks.
:::

## 11.1 The problem

Without paging, an engine must reserve `max_model_len` of KV per running sequence, because the cache must be contiguous and the sequence might grow that long. A request that generates 50 tokens holds memory for 2,048 (or 128k). Three kinds of waste appear: **reserved-but-unused**, **internal fragmentation**, and **external fragmentation** between freed regions.

## 11.2 The idea

Split the cache into blocks of B tokens. Each sequence owns a **block table** — a list of block ids, not a contiguous range. Growing by one token allocates a new block only when the last one is full. Two sequences can point at the same block (copy-on-write), which is how a shared prompt is stored once.

## 11.3 Measured

@file labs/lab35_paged_attention.py
@lab lab35_paged_attention

32 realistic requests: contiguous reservation wasted **81%** of KV memory; the paged allocator wasted **1.9%**. Internal fragmentation with paging is bounded by `B−1` tokens per sequence, no matter how large `max_model_len` is.

## 11.4 What vLLM does with it (read from v0.29.0 source)

|Feature|Default|What it means for you|
|---|---|---|
|`enable_prefix_caching`|**True**|identical prompt prefixes reuse cached blocks across requests|
|`gpu_memory_utilization`|**0.92**|fraction of VRAM the engine may use; the rest is yours to leave free|
|`cache_dtype`|`auto`|fp8 variants exist (`fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp8_ds_mla`, ...) to halve KV memory|
|`block_size`|derived per model|can be set explicitly; must suit the attention backend|
|preemption|scheduler|when blocks run out, sequences are preempted (recomputed or swapped) rather than the server crashing|

Consequences you can feel: shared system prompts become nearly free; `n>1` sampling and beam search stop multiplying KV; throughput degrades gracefully under overload instead of OOM-ing.

## 11.5 Common mistakes

- Setting `gpu_memory_utilization` to 0.99 and then being surprised by allocator failures during long prefills.
- Assuming prefix caching works across different models, LoRA adapters or rope settings. It does not — the cache is keyed to the exact configuration.
- Expecting paging to reduce the **total** KV needed by a single long sequence. It removes waste; it does not compress anything (that is chapters 8-10 and fp8).
- Benchmarking with a hot prefix cache and reporting it as cold-start throughput.

## 11.6 Beyond vLLM

The same idea appears in every serious engine (SGLang's RadixAttention generalises prefix sharing to a tree; TensorRT-LLM has paged KV too). Newer directions — chunked prefill, prefill/decode disaggregation, KV offloading to CPU or NVMe, and cross-request KV reuse — are all built on top of a paged cache.

:::check
- I can explain the block table in one sentence.
- I know vLLM's real defaults for prefix caching and memory utilisation.
- I ran lab 35 and saw the waste numbers myself.
:::
