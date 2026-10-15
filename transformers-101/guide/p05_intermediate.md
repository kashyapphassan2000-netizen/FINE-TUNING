# PART 5 — INTERMEDIATE (Topics 4, 5, 6 + the full block)

Part 4 used one attention "head" with toy sizes. Now you build the real thing: batched **multi-head** attention, **masks**, **positional encodings** (including RoPE, used by almost every 2026 LLM), and the complete **Transformer block** that Part 6 stacks into whole models.

## SECTION 5.0 — Create the Part 5 notebook

**WHERE:** Google Colab
**GOAL:** Fresh notebook with imports.

**Step 1.** Click **File > New notebook in Drive**.

**Step 2.** Rename it to `T101_Part5_Intermediate.ipynb`.

**Step 3.** Click **Runtime > Change runtime type > T4 GPU > Save** (CPU also works for Part 5).

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **5.0-A**:

{{code:5.0-A}}

**Step 6.** Press **Shift+Enter**.

> **You should see:**
{{out:5.0-A}}

**CHECKPOINT:** a torch version and device are printed.

---

## 5.1 Topic 4 — Multi-head attention

**Concept.** One attention head can only express one "kind of relevance" per layer. **Multi-head attention** splits the model width `d_model` into `h` smaller heads (e.g. 512 = 8 × 64). Each head has its own Q/K/V projections, attends independently, and the head outputs are concatenated and mixed by an output matrix `W_O`. It costs about the same as one big head but lets the layer track several relationships at once (syntax, coreference, nearby words…). Every Transformer uses it.

![Multi-head attention](diagrams/d05_multi_head.svg)

### SECTION 5.1.1 — Build `MultiHeadAttention`

**WHERE:** Colab, `T101_Part5_Intermediate.ipynb`
**GOAL:** A batched multi-head attention module you will reuse in Parts 5 and 6.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.1-A**:

{{code:5.1-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.1-A}}

**Line by line**

- `assert d_model % n_heads == 0` — each head gets an equal slice, so `d_model` must divide evenly.
- `W_q, W_k, W_v` are each `d_model × d_model`. This is the same as having `h` separate `d_model × d_head` matrices side by side — one big matrix is faster on a GPU.
- `split_heads`: `view(B, T, h, d_head)` cuts the last axis into heads; `transpose(1, 2)` moves heads in front of tokens so every head is processed like a separate batch item.
- `x_kv=None` — if you pass only `x_q`, keys and values come from the same input (**self**-attention). Passing a *different* `x_kv` turns this exact class into **cross**-attention (Topic 10).
- `math.sqrt(self.d_head)` — scale by the **head** size, not `d_model`.
- `transpose(1, 2).contiguous().view(B, T, H*Dh)` glues the heads back together. `.contiguous()` is needed because `transpose` only changes how memory is read; `view` needs memory laid out in order.
- Parameters: 4 matrices × (32×32 weights + 32 biases) = 4 × 1,056 = **4,224**.

**CHECKPOINT:** weights shape is `(2, 4, 6, 6)` = (batch, heads, query, key).

### SECTION 5.1.2 — Trace every shape

**WHERE:** same notebook
**GOAL:** Watch the tensor change shape at each step (the #1 skill for debugging Transformers).

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.1-B**:

{{code:5.1-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.1-B}}

| Step | Shape | Plain words |
|---|---|---|
| 1 | (B, T, D) | one 32-number vector per token |
| 2 | (B, T, H, D/H) | each vector cut into 4 pieces of 8 |
| 3 | (B, H, T, D/H) | group by head |
| 4 | (B, H, T, T) | each head: every token vs every token |
| 5 | (B, H, T, D/H) | each head: blended values |
| 6 | (B, T, D) | heads glued back to 32 numbers per token |

**CHECKPOINT:** the last shape equals the first shape — attention never changes `(B, T, D)`.

### SECTION 5.1.3 — Prove it equals PyTorch's official layer

**WHERE:** same notebook
**GOAL:** Copy your weights into `nn.MultiheadAttention` and check the outputs match.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.1-C**:

{{code:5.1-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.1-C}}

**Line by line:** PyTorch stores Q, K and V weights stacked in one `in_proj_weight` of shape `(3·D, D)`, so we `torch.cat` ours in the order Q, K, V. `torch.no_grad()` lets us overwrite parameters without recording gradients. `batch_first=True` makes PyTorch use `(B, T, D)` like we do.

**CHECKPOINT:** `True`. You have now written production-equivalent multi-head attention.

**Common mistake at this level — `view` instead of `transpose` (or the reverse).**
Writing `x.view(B, H, T, d_head)` directly *runs* but scrambles tokens across heads. *How to spot it:* the comparison in 5.1-C prints `False`, or a model trains but stays bad. Always `view(B, T, H, d_head)` **then** `transpose(1, 2)`.

---

## 5.2 Topic 5 — Masked multi-head attention

**Concept.** A **mask** forbids some attention links by setting their scores to −∞ before softmax. Two kinds matter:
(1) the **causal (look-ahead) mask** — when a model learns to predict the next word, position *t* must not see positions after *t*, otherwise it just copies the answer; this is the "masked multi-head attention" inside every decoder;
(2) the **padding mask** — batches contain sentences of different lengths padded with a PAD token, and nobody should attend to PAD.

![Masks](diagrams/d06_masks.svg)

### SECTION 5.2.1 — Build the causal mask

**WHERE:** same notebook
**GOAL:** A lower-triangular matrix of 1s.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.2-A**:

{{code:5.2-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.2-A}}

**Meaning:** row = query position, column = key position. Row 2 (third word) may see columns 0, 1, 2 only. `torch.tril` keeps the lower triangle. `.view(1, 1, T, T)` adds batch and head axes so it broadcasts over every batch item and every head.

**CHECKPOINT:** 1s on and below the diagonal, 0s above.

### SECTION 5.2.2 — Prove the mask stops "cheating"

**WHERE:** same notebook
**GOAL:** Change the last word and show that, with the mask, no earlier output changes.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.2-B**:

{{code:5.2-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.2-B}}

**Meaning:** `x2` is `x1` with only the last token replaced. With the causal mask, the first 5 outputs are identical — the future cannot leak backwards. Without the mask, every output changes because every token saw the changed last token. The weight grid shows zeros above the diagonal, and row 0 puts 100% on itself (it can see nothing else).

**CHECKPOINT:** `True`, `True`, `False` in that order.

### SECTION 5.2.3 — Padding mask and combining masks

**WHERE:** same notebook
**GOAL:** Hide PAD tokens and combine with the causal mask.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.2-C**:

{{code:5.2-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.2-C}}

**Meaning:** `(tokens != PAD)` is 1 for real words. Shape `(B, 1, 1, T)` means "for every head and every query, these keys are allowed". Multiplying two masks = logical AND: a key is visible only if it is in the past **and** not padding. In Hugging Face this is the `attention_mask` you pass with every batch.

**CHECKPOINT:** sentence 2 can only ever see its first 2 columns.

> **Watch out:** if an *entire row* is masked (a query that may see nothing), softmax of all −∞ gives `NaN`. Real libraries avoid this by always letting a token see itself or by using a large negative number such as −1e9 instead of −∞.

**Common mistake at this level — forgetting the causal mask while training a GPT.**
*How to spot it:* training loss drops to almost 0 suspiciously fast, but generated text is garbage. The model learned to copy the next token it could see. Re-run test 5.2-B on your model's attention.

---

## 5.3 Topic 6 — Positional encoding

**Concept.** Attention compares every token with every token as an unordered **set**. Shuffle the words and the outputs just shuffle the same way — "dog bites man" and "man bites dog" look identical. **Positional encoding** injects order. Three styles: **sinusoidal** (2017 paper: fixed sine/cosine waves *added* to embeddings), **learned** (GPT-2, BERT: a trainable table of position vectors), and **RoPE** (rotary: *rotates* query and key vectors by an angle proportional to position — used by Llama, Qwen, Mistral, DeepSeek and most 2026 LLMs [14]).

![Positional encoding](diagrams/d07_positions.svg)

### SECTION 5.3.1 — Prove attention is blind to order

**WHERE:** same notebook
**GOAL:** Show that shuffling inputs only shuffles outputs.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.3-A**:

{{code:5.3-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.3-A}}

**Meaning:** `x[:, perm]` reorders the 5 tokens. The output of the shuffled input equals the shuffled output of the original — mathematically, self-attention is *permutation-equivariant*. This is also why rows 0 and 4 ("the" and "the") were identical in 4.3-A.

**CHECKPOINT:** `True`.

### SECTION 5.3.2 — Build sinusoidal positional encoding

**WHERE:** same notebook
**GOAL:** Implement the 2017 formula and look at it.

The formula from the paper [9], for position `pos` and dimension pair `i`:

`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`  `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.3-B**:

{{code:5.3-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.3-B}}

{{fig:5.3-B}}

**Line by line**

- `pos` is a column `[0, 1, …, 99]`; `i` is `[0, 2, 4, …, 62]`.
- `freq = exp(−ln(10000)·i/d)` is the same as `1 / 10000^(i/d)`, written with `exp` for numerical stability. Low dimensions get high frequencies (fast waves), high dimensions get slow waves.
- `pe[:, 0::2] = sin(...)` fills even columns; `pe[:, 1::2] = cos(...)` fills odd columns.
- Position 0 is `[0, 1, 0, 1, …]` because sin(0)=0, cos(0)=1.
- In the heatmap each column is a unique "barcode" for one position. In a model you would do `x = token_embedding + pe[:T]`.

**CHECKPOINT:** table shape `(100, 64)` and the striped heatmap.

### SECTION 5.3.3 — Nearby positions look alike

**WHERE:** same notebook
**GOAL:** Show the encoding carries a sense of distance.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.3-C**:

{{code:5.3-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**

{{fig:5.3-C}}

{{out:5.3-C}}

**Meaning:** the dot product of position 20 with itself is 32 (= 64/2, since sin²+cos² = 1 for each of the 32 pairs). Position 21 is nearly as similar (30.9); position 60 is much less similar (15.4). The model can use this to tell "near" from "far".

**CHECKPOINT:** the curve peaks at the red line and falls away on both sides.

### SECTION 5.3.4 — Build RoPE and prove it encodes *relative* distance

**WHERE:** same notebook
**GOAL:** Implement rotary position embedding and verify its key property.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.3-D**:

{{code:5.3-D}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.3-D}}

**Line by line**

- RoPE treats each pair of numbers `(x1, x2)` as a 2-D point and **rotates** it by `angle = position × freq`. The rotation formula is the standard one: `(x1·cos − x2·sin, x1·sin + x2·cos)`.
- It is applied to **queries and keys only** (not values), *inside* attention, instead of adding anything to the embeddings.
- Because rotating q by angle *a* and k by angle *b* gives a dot product that depends only on *a − b*, the score depends only on the **distance** between tokens: query 3 vs key 1 scores exactly the same as query 30 vs key 28 (−10.1427). Distance 20 gives a different score.
- Rotation never changes a vector's length, so RoPE does not distort the token's content.

**CHECKPOINT:** the two "distance 2" lines show the same number.

**Common mistake at this level — adding positions twice or in the wrong place.**
RoPE is applied to Q and K inside each attention layer; sinusoidal/learned PE is added once to the input embeddings. Mixing them up (e.g. adding a learned PE *and* RoPE by accident, or applying RoPE to V) still trains but wastes capacity. *How to spot it:* read the model's config (`rope_theta`, `max_position_embeddings`) and its forward code before modifying it.

---

## 5.4 Putting it together — one Transformer block

**Concept.** A **block** = multi-head attention (tokens exchange information) + a feed-forward MLP (each token processes what it gathered), each wrapped in **LayerNorm** and a **residual** ("skip") connection: `x = x + Attention(LN(x))`, then `x = x + MLP(LN(x))`. Putting the LayerNorm *before* each sub-layer ("pre-norm") trains far more stably than the original 2017 "post-norm" [46]; every modern model does it. Blocks keep the shape `(B, T, d_model)`, so you can stack as many as you like.

![Transformer block](diagrams/d08_block.svg)

### SECTION 5.4.1 — Build `FeedForward` and `Block`

**WHERE:** same notebook
**GOAL:** The reusable block for Part 6.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **5.4-A**:

{{code:5.4-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:5.4-A}}

**Line by line**

- `FeedForward`: expand to `4 × d_model`, apply GELU (a smooth ReLU), shrink back. It runs on every token **separately** — no mixing between tokens here.
- `x = x + a` — the residual connection. Gradients can flow straight through the `+`, which is why 100-layer models train at all.
- `self.ln1(x)` normalises each token vector to mean 0 / variance 1 before attention.
- Parameter count: attention 4 × (64·64+64) = 16,640; MLP 64·256+256 + 256·64+64 = 33,088; two LayerNorms 2 × 128 = 256 → **49,984**. The MLP holds about **2/3 of a block's parameters** — true for real LLMs too.

**CHECKPOINT:** input and output shapes are both `(2, 10, 64)`. Click **File > Save**.

**Level-up check (Part 5 done):** you have hand-built, verified versions of multi-head attention, causal and padding masks, sinusoidal PE, RoPE, and a full pre-norm Transformer block. Part 6 stacks these into complete models.
