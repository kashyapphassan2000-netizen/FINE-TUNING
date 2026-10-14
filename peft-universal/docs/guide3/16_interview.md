# 16. Fifteen interview questions, with crisp answers

1. **Why is a KV cache needed at all?** Without it, each generated token re-processes the whole sequence: O(n²) work. The cache makes each step O(1) new K/V plus attention over the cache.
2. **What exactly is stored?** Per layer, per KV head: the key and value vectors of every past token. Not Q, not MLP activations.
3. **What is the memory formula?** `2 × layers × kv_heads × head_dim × bytes × tokens_in_flight`.
4. **Prefill vs decode?** Prefill processes the prompt in one compute-bound pass; decode generates token by token and is memory-bandwidth bound.
5. **MHA vs MQA vs GQA?** MHA: one K/V per query head. MQA: one K/V shared by all. GQA: one K/V per group — near-MHA quality at close to MQA cost; the modern default.
6. **What is MLA and why does it exist?** Cache one low-rank latent per token and rebuild per-head K/V on the fly. It makes very-high-head-count models (DeepSeek-V3: 128 heads) cheap to serve at long context.
7. **Why does MLA need a decoupled RoPE dimension?** The rotary rotation does not commute with absorbing the up-projection into the query, so a small rotary key part is cached separately.
8. **What does FlashAttention change?** Not the result: the memory traffic. Tiled computation with online softmax avoids materialising and re-reading the N×N scores.
9. **What does `torch.nn.functional.scaled_dot_product_attention` do for you?** Picks among flash / memory-efficient / cuDNN / math kernels; handles causal masking, scale, dropout and GQA repetition (`enable_gqa`).
10. **When must `is_causal` be False?** During decode with a single query token against a longer cache.
11. **What problem does PagedAttention solve?** Fragmentation and over-reservation: KV is stored in fixed blocks addressed by a per-sequence block table, so waste is bounded and blocks can be shared copy-on-write.
12. **How does prefix caching save money?** Identical prompt prefixes map to the same blocks, so shared system prompts are stored and computed once. Enabled by default in vLLM 0.29.
13. **Why can't attention tell word order by itself?** It is permutation-equivariant; positional encoding (absolute, ALiBi, RoPE) supplies order.
14. **How does RoPE work and why is it popular?** It rotates Q and K by an angle proportional to position, so scores depend on relative distance; no parameters, works with caching, and extends with scaling methods.
15. **How do you extend context to 128k?** Choose a scaling method (linear / NTK / YaRN / llama3), fine-tune on long data, re-evaluate with retrieval-style long-context tests, and re-check KV memory — it grows linearly with context.
