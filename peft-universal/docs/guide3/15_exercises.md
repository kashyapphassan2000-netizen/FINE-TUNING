# 15. Ten exercises, with solutions

1. **A model has 48 layers, 8 KV heads, head_dim 128, served in bf16. What is the KV cost per token, and for 64 users at 4,096 tokens?**
   *Solution:* `2 × 48 × 8 × 128 × 2 = 196,608 bytes = 192 KB/token`. 64 × 4,096 = 262,144 tokens → `192 KB × 262,144 = 48 GB`.

2. **The same model switched to MQA (1 KV head). How much does one token cost now, and what did you trade?**
   *Solution:* `2 × 48 × 1 × 128 × 2 = 24 KB/token`, an 8× cut. Trade: fewer distinct K/V subspaces, reported quality loss; GQA exists as the middle ground.

3. **Without a cache, how many token-forwards to generate 1,000 tokens? With one?**
   *Solution:* without: `1000 × 1001 / 2 = 500,500`. With: `1,000`. Ratio ≈ 500×.

4. **Why does a decode step pass `is_causal=False` to SDPA?**
   *Solution:* there is a single query token and it may attend to every cached key. `is_causal=True` would apply a triangular mask against the cache length and hide the past.

5. **Your engine reserves `max_model_len=8192` per sequence but average usage is 600 tokens. What fraction of KV memory is wasted, and what fixes it?**
   *Solution:* `1 − 600/8192 ≈ 92.7%`. PagedAttention: blocks allocated on demand, waste bounded by `block_size − 1` per sequence (lab 35 measured 81% → 1.9%).

6. **Compute DeepSeek-V3's KV per token and compare with a same-depth GQA model with 8 KV heads and head_dim 128.**
   *Solution:* MLA: `61 × (512 + 64) × 2 = 70,272 B ≈ 68.6 KB`. GQA: `2 × 61 × 8 × 128 × 2 = 249,856 B ≈ 244 KB`. MLA is ~3.6× cheaper here, and far cheaper than the 128-head MHA alternative.

7. **Why is FlashAttention "exact" if it never builds the N×N matrix?**
   *Solution:* softmax can be rescaled: keep running max `m` and sum `l`; when a block raises the max, multiply the partial output and `l` by `exp(m_old − m_new)`. Only float rounding differs (lab 32: 2.4e-07).

8. **You extend a RoPE model from 8k to 64k context by setting a linear scaling factor of 8 at serving time. What is wrong?**
   *Solution:* the model was never trained at those interpolated angles. Scaling needs fine-tuning on long examples plus long-context evaluation (needle tests), not just a config flag. Also, changing rope invalidates any warm prefix cache.

9. **Your server has 8 KV heads and you set tensor parallel = 16. What happens to KV memory?**
   *Solution:* KV heads cannot be split 16 ways; they are replicated, so per-GPU cache stops shrinking and total KV memory duplicates. Keep `TP ≤ kv_heads`.

10. **A Gemma-3-style model uses 5 sliding layers (window 4,096) per 1 full layer. Why does the chapter-3 formula overestimate its cache at 128k context?**
    *Solution:* sliding layers stop growing at the window, so only the full-attention layers scale with context. Roughly `(5 × 4096 + 1 × 131072)/6` tokens' worth per layer-set instead of 131,072.
