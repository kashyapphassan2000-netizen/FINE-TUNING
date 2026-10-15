# PART 7 — ELITE: how top labs and researchers build Transformers in 2026

## 7.0 What changed since 2017 — the modern recipe

Your Part 6 MiniGPT is essentially GPT-2 (2019). Every serious open model since Llama (2023) changed the same handful of parts. Interviewers at AI labs expect you to know *what* changed and *why*.

| Part | 2017 / GPT-2 (your MiniGPT) | 2023–2026 standard (Llama 3, Qwen3, Mistral, DeepSeek, Gemma) | Why it changed |
|---|---|---|---|
| Normalisation | LayerNorm, post-norm (2017) / pre-norm (GPT-2) | **RMSNorm**, pre-norm [18]; often **QK-norm** on queries/keys (Qwen3, OLMo 2) | Cheaper, equally good; QK-norm stops attention logits exploding |
| Positions | Sinusoidal / learned table | **RoPE** [14] (+ YaRN-style scaling for long context) | Relative distance, extends to longer contexts |
| MLP | Linear → ReLU/GELU → Linear | **SwiGLU** gated MLP [17] | Better quality per parameter |
| Attention heads | Full multi-head (MHA) | **GQA** [15] (Llama 3, Qwen3), **MQA** [16], **MLA** (DeepSeek-V3 [20]) | Shrinks the KV cache → cheaper long-context inference |
| Attention kernel | Plain matmul + softmax | **FlashAttention** 2/3/4 [19], PyTorch SDPA [34] | Never materialise the T×T matrix; FA4 targets NVIDIA Blackwell (Mar 2026) |
| MLP capacity | Dense: every token uses every weight | **Mixture-of-Experts** (DeepSeek-V3, Qwen3-MoE, Kimi K2) [38][20] | Huge total parameters, small active parameters per token |
| Attention span | Full attention every layer | **Sliding-window / local–global** mixes (Gemma 3, Mistral), **sparse** attention (DeepSeek-V3.2), **hybrid linear attention** (Qwen3-Next: 3 Gated DeltaNet layers per 1 gated-attention layer; Kimi Linear) [31] | Escape the T² cost for very long contexts |
| Output layer | Separate or tied to embedding | Tied in small models, untied in large | Saves parameters in small models |
| Generation | Recompute everything each step | **KV cache** (+ paged attention in vLLM) | Turns O(T²) generation into O(T) work per step |

Sebastian Raschka's *LLM Architecture Gallery* and *The Big LLM Architecture Comparison* track these choices model by model [25]; read them after this Part.

![Attention variants](diagrams/d13_attention_variants.svg)

## 7.1 Who defines the top level (names to know)

| Person / org | Contribution you should be able to explain |
|---|---|
| Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google, 2017) | *Attention Is All You Need* — the Transformer [9] |
| Noam Shazeer | Multi-query attention [16], SwiGLU/GLU variants [17], sparsely-gated MoE |
| Jianlin Su et al. | RoPE (RoFormer) [14] |
| Tri Dao (Princeton / Together AI) | FlashAttention 1–4 [19]; with Albert Gu, Mamba state-space models [32] |
| DeepSeek | MLA + fine-grained MoE + multi-token prediction (DeepSeek-V3) [20]; sparse attention (V3.2) |
| Alibaba Qwen team | Qwen3 dense/MoE; Qwen3-Next hybrid linear + gated attention [31] |
| Anthropic interpretability (Chris Olah et al.) | *Mathematical Framework for Transformer Circuits*, **induction heads** [21][22] |
| Neel Nanda | TransformerLens library [26]; mechanistic interpretability tutorials |
| Andrej Karpathy | nanoGPT, *Let's build GPT* video, **nanochat** (full ChatGPT-style pipeline for ~$100) [24] |
| Keller Jordan et al. | **modded-nanogpt** speedrun (GPT-2-124M quality in minutes), Muon optimizer |
| Sebastian Raschka | *Build a Large Language Model (From Scratch)* + repo [25] |
| Stanford CS336 (Percy Liang, Tatsunori Hashimoto) | *Language Modeling from Scratch* course: build, scale, optimise LMs [42] |
| AI2 (OLMo) | Fully open models: data, code, checkpoints — best for research reproductions |

**How elite people actually work** (copy these habits):

1. **Read code, not only papers.** Papers omit details; the reference implementation (Hugging Face `modeling_*.py`, nanoGPT, the paper's repo) is the truth.
2. **Reproduce a baseline first, then change one thing.** Same data, same compute budget, same seed policy. Report validation loss, not vibes.
3. **Measure cost, not only quality:** tokens/second, GPU memory, KV-cache size. Most 2024–2026 architecture work is about cost.
4. **Test mechanisms directly** (interpretability): don't trust a pretty heatmap; define a score and measure it across all heads (you do this in 7.4).

---

## SECTION 7.2 — Build a "Llama/Qwen-style" decoder: RMSNorm + RoPE + SwiGLU + GQA + SDPA

**WHERE:** Google Colab — new notebook `T101_Part7_Elite.ipynb`, **T4 GPU**
**GOAL:** Upgrade MiniGPT to the 2026 recipe and train it on the same data and budget.

**Step 1.** Click **File > New notebook in Drive**.

**Step 2.** Rename it to `T101_Part7_Elite.ipynb`.

**Step 3.** Click **Runtime > Change runtime type**, choose **T4 GPU**, click **Save**.

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **7.0-A** (imports + the same Shakespeare data pipeline as Part 6):

{{code:7.0-A}}

**Step 6.** Press **Shift+Enter**.

> **You should see:** `torch ... | device: cuda` then
> `data ready: 1,003,854 train tokens, vocab 65`

**Step 7.** Click **+ Code**.

**Step 8.** Paste cell **7.1-A** (the four modern components):

{{code:7.1-A}}

**Step 9.** Press **Shift+Enter**.

> **You should see:**
{{out:7.1-A}}

**Line by line**

- **RMSNorm**: divide each vector by its root-mean-square, then scale by a learned weight. Unlike LayerNorm it does not subtract the mean and has no bias — fewer operations, same stability.
- **`rope_tables`** pre-computes cos/sin for every position once. **`apply_rope`** is the rotation you proved in 5.3-D, written with `stack(...).flatten(-2)` so it interleaves the rotated pairs back into place.
- **GQAttention**: `wq` produces `n_heads` query heads, but `wk`/`wv` produce only `n_kv_heads` (here 8 query heads, 2 KV heads). `repeat_interleave` lets each group of 4 query heads share one K/V head. Same quality idea as full MHA, ¼ of the KV memory.
- The `cache` dictionary stores past K and V (the **KV cache**, used in Section 7.3). New K/V are concatenated on the time axis (`dim=2`).
- `F.scaled_dot_product_attention(..., is_causal=(T > 1))` is PyTorch's fused attention. On Ampere/Hopper GPUs it dispatches to FlashAttention; on the T4 it uses the memory-efficient kernel. `is_causal=True` applies the causal mask without building it. With one new token (`T == 1`) no mask is needed — it may see the whole cache.
- **SwiGLU**: two parallel projections; one goes through SiLU and acts as a **gate** that multiplies the other. Hidden size ≈ 8/3 × d (rounded to 64) keeps the parameter count close to a 4× GELU MLP.
- **ModernBlock** is the same pre-norm residual pattern as Part 5, with the new parts plugged in.

**Step 10.** Click **+ Code**.

**Step 11.** Paste cell **7.1-B** (the full model):

{{code:7.1-B}}

**Step 12.** Press **Shift+Enter**.

> **You should see:**
{{out:7.1-B}}

**Line by line**

- There is **no `pos_emb`** — positions live entirely in RoPE inside attention.
- `self.head.weight = self.tok_emb.weight` — **weight tying**: the output layer reuses the embedding table (saves 16k parameters here, ~0.5–1 B in mid-size LLMs).
- `self.apply(self._init_weights)` starts every Linear and Embedding weight as small random numbers (standard deviation 0.02, the GPT-2 recipe). **This line matters:** while testing this guide, the model *without* it started at a loss of **237** instead of ~4.2, because PyTorch's default embedding init (std 1.0) is shared with the output layer through weight tying and produces huge logits. Initialisation is part of the architecture.
- `register_buffer(..., persistent=False)` stores the RoPE tables on the GPU without saving them in checkpoints.
- `start_pos` tells RoPE the absolute position of the first token being processed — essential when you feed one token at a time with a cache.
- `generate(use_cache=False)` re-runs the whole prefix every step (like MiniGPT). `generate(use_cache=True)` runs the prompt once ("prefill"), then feeds **only the newest token** each step ("decode") — this is how every production inference server works.

**Step 13.** Click **+ Code**.

**Step 14.** Paste cell **7.1-C** (training with modern practice):

{{code:7.1-C}}

**Step 15.** Press **Shift+Enter** and wait (≈3–6 minutes on T4).

> **You should see:** loss lines every 300 steps, finishing around **val ≈ 1.45–1.55** on T4 after 3,000 steps. Compare with your Part 6 MiniGPT at the same step count. On the CPU build machine (both runs shortened to 600 steps) the result was **MiniGPT val 1.714 vs ModernGPT val 1.505 — with 12% fewer parameters** (2.84 M vs 3.23 M). Honest caveat: the two runs also differ in optimiser settings and LR schedule, so this is not a pure architecture ablation — fixing that is Project 2 in Part 11. Reference log from that shortened run:
{{out:7.1-C}}

**Line by line**

- `torch.autocast(..., dtype=torch.float16)` = **mixed precision**: matrix multiplies run in 16-bit (≈2× faster on T4 tensor cores, half the activation memory); sensitive operations stay 32-bit. On A100/H100 use `torch.bfloat16` and drop the scaler.
- `GradScaler` multiplies the loss before `backward()` so tiny float16 gradients don't round to zero, then un-scales before the optimizer step. `scaler.unscale_` must come before `clip_grad_norm_`.
- `AdamW(betas=(0.9, 0.95), weight_decay=0.1)` — the betas and decay used by most LLM training recipes.
- `OneCycleLR(pct_start=0.1)` = **warm-up** for the first 10% of steps, then cosine-like decay. Warm-up avoids early instability; decay squeezes out the last bit of loss.

| If you see... | It means... | Do this next |
|---|---|---|
| `loss = nan` | float16 overflow | Make sure you used `GradScaler`; or set `use_amp = False` |
| val worse than MiniGPT by > 0.1 | A bug (often RoPE or the mask) | Re-run Part 5.3-D test; check `is_causal=(T > 1)` |
| `RuntimeError ... size of tensor a (32) must match ... (16)` | `cos/sin` built with the wrong head size | `rope_tables(max_len, d_model // n_heads)` |
| loss before training is in the hundreds | Weight init missing with tied embeddings | Make sure `self.apply(self._init_weights)` is in `__init__` (7.1-B) |

**CHECKPOINT:** training finishes, final val loss < 1.8.

**Common mistake at this level — benchmarking unfairly.** Comparing two architectures with different step counts, learning rates or token budgets proves nothing. *How to spot it:* you cannot state the exact budget for each run. Elite practice: fixed tokens, tuned LR for each, ≥2 seeds.

---

## SECTION 7.3 — KV cache: measure the speed-up and the memory bill

**WHERE:** same notebook
**GOAL:** Prove the KV cache gives identical text, measure tokens/second, then compute KV memory for real models.

![KV cache](diagrams/d12_kv_cache.svg)

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **7.2-A**:

{{code:7.2-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see** something like this (speed numbers depend on hardware; the ratio grows with longer outputs and bigger models):
{{out:7.2-A}}

**Line by line**

- One short warm-up call (`generate(prompt, 5)`) so first-call setup does not distort timing.
- `torch.cuda.synchronize()` — GPU work is asynchronous; without it you would time only the *launch* of the work.
- `torch.equal(slow, fast)` — greedy decoding with and without cache must give the **same** tokens. If this is `False`, your cache or RoPE offsets (`start_pos`) are wrong.
- We generate 110 tokens so the total (116) stays under the 128-token training window; beyond it the no-cache path crops context and the two would legitimately differ.
- The sample text loops ("of the state of the state…") because **greedy** decoding always picks the single most likely token; small models fall into repetition. That is why chat products sample with temperature/top-p (Part 6.3-E) — the KV cache works identically with sampling.

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **7.2-B**:

{{code:7.2-B}}

**Step 6.** Press **Shift+Enter**.

> **You should see:**
{{out:7.2-B}}

**Meaning:** the formula is `2 (K and V) × layers × kv_heads × head_dim × tokens × bytes`. For an 8B-class model a single 32k-token chat needs **17 GB** of cache with full MHA but **4.3 GB** with GQA — the reason Llama 3 and Qwen3 use GQA and DeepSeek invented MLA. KV-cache size, not weights, is often what limits how many users one GPU can serve. This is a top interview question for inference/serving roles.

**CHECKPOINT:** `identical text ... True`, and the cache version is faster.

**Common mistake — forgetting position offsets with a cache.** If you apply RoPE to the new token as position 0 every step, generation slowly turns to garbage. *How to spot it:* `identical text` prints `False`. Fix: pass `start_pos = current_length - 1` (as in `generate`).

---

## SECTION 7.4 — Mechanistic interpretability: find the induction heads in GPT-2

**Concept.** Anthropic's 2022 research [21] showed that small and large Transformers develop **induction heads**: a two-layer circuit that implements "if the pattern `A B` appeared before and I now see `A` again, predict `B`". It is believed to be a core mechanism of **in-context learning** (why LLMs can copy names, follow examples in a prompt, etc.). The measurement trick: feed a sequence of **random tokens repeated twice**. On the first copy nothing is predictable; on the second copy a model with induction heads predicts almost perfectly, and those heads attend from each token to the token *right after its previous occurrence*.

**WHERE:** same notebook (downloads GPT-2 small, ~550 MB)
**GOAL:** Measure in-context learning and score all 144 heads of GPT-2.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **7.3-A**:

{{code:7.3-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:7.3-A}}

**Line by line**

- `torch.randint(1000, 10000, (1, L))` — 50 random token IDs (avoiding the most common tokens). The full input is `[BOS] + random + random` = 101 tokens.
- `log_softmax(logits[:-1])` and `gather(seq[1:])` — the log-probability the model gave to the **actual next token** at every position; minus that is the loss per token.
- On the first copy the loss is ~10 (pure guessing among 50k tokens). On the second copy it collapses to <1: the model is copying from context it has seen only once. That is in-context learning, measured.

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **7.3-B**:

{{code:7.3-B}}

**Step 6.** Press **Shift+Enter**.

> **You should see:** a 12 × 12 heatmap with a few bright cells in the middle layers, and a ranking like:

{{fig:7.3-B}}

{{out:7.3-B}}

**Line by line**

- For a query at position `i` in the second copy, its previous occurrence is at `i − L`, so the token that came **next** last time is at `i − L + 1`. `att[:, :, q_pos, k_pos]` reads the attention from every query to exactly that key, for all layers and heads at once; `.mean(-1)` averages over the 50 positions → one **induction score** per head.
- A score near 1 means the head does almost nothing except "look at what came after my last occurrence". Heads in layers 5–7 of GPT-2 small (e.g. 5.1, 5.5, 6.9, 7.10) are the well-known induction heads found by the interpretability community — you just rediscovered them with ~15 lines of code.

**CHECKPOINT:** second-copy loss far below first-copy loss, and at least a few heads score > 0.5.

**Go further (elite):** install TransformerLens [26] (`pip install transformer_lens`) and **ablate** (zero out) your top induction heads with a hook, then re-measure the second-copy loss. If it jumps up, you have shown the heads are *causally* necessary, not just correlated — this is the standard of evidence in interpretability research.

**Common mistake at this level — off-by-one in position bookkeeping.** Using `k_pos = q_pos − L` scores "previous-occurrence" heads instead of induction heads. *How to spot it:* different heads light up and scores are lower. Write the indices out on paper for a 3-token example first.

---

## 7.5 The frontier: what to study next (and what to read)

| Direction | One-line idea | Start here |
|---|---|---|
| FlashAttention 2/3/4 | Tile Q, K, V into fast on-chip memory; never store the T×T matrix | Dao et al. [19]; FA4 paper arXiv:2603.05451 |
| Mixture-of-Experts | A router sends each token to 2–8 of many expert MLPs | Switch Transformer [38]; DeepSeek-V3 [20] |
| MLA | Compress K/V into a small latent; expand on the fly | DeepSeek-V2/V3 [20] |
| Hybrid / linear attention | Most layers use O(T) recurrent "linear attention" (DeltaNet, Mamba-2), a few keep full softmax attention | Raschka hybrid-attention gallery [25]; Qwen3-Next; Mamba [32] |
| Gated attention | A learned sigmoid gate on the attention output; removes attention sinks, stabilises training | Qiu et al. 2025, *Gated Attention for LLMs* (arXiv:2505.06708) (verify) |
| Long context | RoPE scaling (YaRN), sliding windows, sparse attention | DeepSeek-V3.2 sparse attention; Gemma 3 report |
| Interpretability | Circuits, sparse autoencoders, attribution graphs | transformer-circuits.pub [21][22] |
| Training efficiency | Speedruns, Muon optimizer, μP | modded-nanogpt; nanochat [24] |

**Your elite-level next build:** clone `karpathy/nanochat` [24] and read `gpt.py` side by side with your Part 7 model — you will now recognise RoPE, RMSNorm/QK-norm, GQA and the KV cache line by line. Then reproduce one ablation (e.g. GQA 2 vs 8 KV heads) on Tiny Shakespeare with a fixed budget and write it up on GitHub. That write-up is a portfolio piece that very few applicants have.
