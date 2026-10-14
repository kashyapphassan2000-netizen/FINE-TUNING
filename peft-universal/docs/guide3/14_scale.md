# 14. Putting it together at any scale, including trillion-parameter models

## 14.1 The two budgets on every GPU

$$ VRAM = weights + KV cache + activations/workspace
Weights are fixed by the model and your dtype. KV grows with users and context. Whatever is left over is your concurrency.

|Model class|Weights (bf16)|KV per token|What dominates|
|---|---|---|---|
|1-8B dense, GQA|2-16 GB|~128 KB|weights on small cards, KV at long context|
|70B dense, GQA|140 GB|327 KB|weights (multi-GPU), then KV|
|405B dense, GQA|810 GB|516 KB|weights: multi-node|
|671B MoE + MLA (DeepSeek-V3)|~1.3 TB|68.6 KB|weights hugely; KV is small by design|
|1T MoE + MLA (Kimi-K2 shape)|~2 TB|similar to above|weights; MoE keeps compute low|

The pattern at the top end is deliberate: **MoE for cheap compute, MLA for a cheap cache, many GPUs for the weights.** A hypothetical 10T model with MHA would be unservable at long context regardless of how many GPUs you bought — the cache alone would eat them.

## 14.2 How the work is split across GPUs

|Parallelism|Splits|Effect on KV|
|---|---|---|
|tensor (TP)|each layer's matrices across GPUs|KV splits across GPUs **if** `TP ≤ kv_heads`; above that it is replicated|
|pipeline (PP)|layers across GPUs/nodes|each GPU holds the KV of its own layers|
|expert (EP)|MoE experts across GPUs|no direct KV effect|
|data|whole replicas|KV per replica|

Practical rule from chapter 9: with 8 KV heads, TP 8 is the natural limit before the cache starts duplicating.

## 14.3 The serving techniques that follow from this book

|Technique|What it does|Built on|
|---|---|---|
|continuous batching|new requests join the running batch every step|the prefill/decode split (ch 1)|
|chunked prefill|long prompts processed in chunks so decode is not starved|same|
|prefix caching|shared prompts stored once|paging (ch 11)|
|fp8 KV cache|halves cache bytes|ch 3|
|quantized weights (fp8/MXFP4)|halves or quarters weights|frees VRAM for KV|
|prefill/decode disaggregation|run the two phases on different machines|their different bottlenecks|
|speculative decoding|a small model drafts, the big one verifies|decode being bandwidth-bound|

:::warn Honest limits of this chapter
None of the throughput claims here were benchmarked in this book (no GPU). The memory arithmetic is exact and reproducible with `peftkit.kv`; the performance statements are structural (what is compute-bound vs bandwidth-bound) and should be measured on your hardware before you promise anyone numbers.
:::

## 14.4 A sizing procedure that works for any model, now or in five years

1. Read the config: `num_key_value_heads`, `head_dim`, `num_hidden_layers`, `kv_lora_rank`, `layer_types`, `rope_parameters`.
2. `python -m peftkit.kv <model> --context <your context> --gpu-gb <card> --gpus <n>` → bytes/token, KV budget, concurrent sequences.
3. `python -m peftkit.plan <model>` (Volume 1) → weights, total memory, GPU count, multi-node flag.
4. Decide: shorten context, fp8 KV, more GPUs, or a different base model. In that order of cheapness.
5. Start the server and compare its reported maximum concurrency with your estimate. Trust the server.
6. Load-test before promising latency or cost per token.

:::check
- I can say which of weights or KV dominates for my deployment.
- My TP size respects the KV head count.
- I verified my estimate against what the engine reported at start-up.
:::
