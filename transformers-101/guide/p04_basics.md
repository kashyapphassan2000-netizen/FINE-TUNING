# PART 4 — BASICS (Topics 1, 2, 3)

You will build one Colab notebook, `T101_Part4_Basics.ipynb`, cell by cell. Every cell builds on the previous ones, so run them in order. A ready-made copy is in the repository at `transformers-101/notebooks/T101_Part4_Basics.ipynb`.

## SECTION 4.0 — Create the Part 4 notebook

**WHERE:** Google Colab in Chrome
**GOAL:** A fresh notebook with PyTorch loaded and a GPU (or CPU) chosen.

**Step 1.** Click **File > New notebook in Drive**.

**Step 2.** Click the title and rename it to `T101_Part4_Basics.ipynb`, then press **Enter**.

**Step 3.** Click **Runtime > Change runtime type**, choose **T4 GPU**, click **Save**. (Part 4 also runs fine on CPU.)

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **4.0-A**:

{{code:4.0-A}}

**Step 6.** Press **Shift+Enter**.

> **You should see** (on Colab: `cuda` and a `+cu12x` version):
{{out:4.0-A}}

**What the non-obvious lines mean**

- `torch.nn` holds ready-made layers; `torch.nn.functional` (as `F`) holds plain functions such as `softmax` and `one_hot`.
- `torch.manual_seed(42)` makes random numbers repeatable, so your numbers match this guide.
- `device = ...` picks the GPU if one is attached.

**CHECKPOINT:** the cell printed a torch version and a device.
*If not:* `ModuleNotFoundError: torch` means you are not in Colab or not in your `.venv` — see Section 2.5 Step 8.

---

## 4.1 Topic 1 — Embeddings: from discrete to continuous space

**Concept (what, why, when).**
A computer cannot do maths on the word "cat", so we first give every token an **ID number** (discrete). IDs carry no meaning — ID 0 is not "closer" to ID 1 than to ID 4. An **embedding** replaces each ID with a learned **vector of real numbers** (continuous), so that words used in similar ways end up **close together in space**. Every Transformer starts with this lookup; its size is `vocab_size × d_model` (GPT-2: 50,257 × 768).

![Embeddings diagram](diagrams/d02_embeddings.svg)

### SECTION 4.1.1 — Turn words into IDs

**WHERE:** Colab, `T101_Part4_Basics.ipynb`
**GOAL:** Build the simplest possible tokenizer: one ID per unique word.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.1-A**:

{{code:4.1-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.1-A}}

**Line by line:** `sorted(set(words))` keeps each unique word once in alphabetical order. `stoi` = "string to integer" (word → ID); `itos` is the reverse. "the" appears twice so its ID `4` appears twice.

**CHECKPOINT:** `ids` has 6 numbers and contains `4` twice.

### SECTION 4.1.2 — See why IDs and one-hot vectors are not enough

**WHERE:** same notebook
**GOAL:** Prove that one-hot vectors say every word is equally different from every other word.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.1-B**:

{{code:4.1-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.1-B}}

**Meaning:** a one-hot vector is all zeros with a single 1 at the word's ID. The dot product of *any two different* words is 0, so "cat" is exactly as unrelated to "mat" as to "the". With a 50,000-word vocabulary each one-hot vector also has 50,000 numbers — huge and empty.

**CHECKPOINT:** `cat . mat = 0.0` and `cat . cat = 1.0`.

### SECTION 4.1.3 — Build an embedding table and prove it is just a lookup

**WHERE:** same notebook
**GOAL:** Create `nn.Embedding`, use it, and prove it equals "one-hot × matrix".

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.1-C**:

{{code:4.1-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.1-C}}

**Line by line**

- `nn.Embedding(num_embeddings=5, embedding_dim=4)` creates a 5 × 4 table of random numbers. Row *i* is the vector for ID *i*. These numbers are **trainable** — training moves them until similar words sit close together.
- `emb(torch.tensor(ids))` picks one row per ID → shape `(6, 4)`: 6 words, 4 numbers each.
- Rows 0 and 4 of `X` are identical because both are the word "the" (ID 4).
- `grad_fn=<EmbeddingBackward0>` means PyTorch remembers how this was computed so it can learn later.
- `one_hot @ emb.weight` gives exactly the same result: **an embedding layer is a one-hot vector multiplied by a weight matrix**, done efficiently as a lookup. This is a classic interview question.

**CHECKPOINT:** `X shape: (6, 4)` and `lookup == one_hot @ W ? True`.

### SECTION 4.1.4 — Measure meaning in a real trained model (GPT-2)

**WHERE:** same notebook (needs internet; downloads ~550 MB once)
**GOAL:** See that trained embeddings put related words close together.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.1-D**:

{{code:4.1-D}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.1-D}}

**Line by line**

- `AutoTokenizer` / `AutoModel` download GPT-2's tokenizer and network from Hugging Face. `attn_implementation="eager"` asks for the plain attention code so we can read the attention weights later (the fast default does not return them).
- `get_input_embeddings().weight` is GPT-2's embedding table: **50,257 tokens × 768 numbers**.
- `tok.encode(" " + word)` — GPT-2 tokens include the leading space, so " king" is one token but "king" at the start of text may be a different token.
- The `assert` stops you if a word is split into several tokens (then it has no single vector).
- `F.cosine_similarity` gives a number between −1 and 1; higher = more similar direction.

Related pairs (king–queen 0.66, happy–glad 0.62, Paris–London 0.58) score far higher than unrelated pairs (≈0.23–0.26). Nobody programmed this — it emerged from training on text.

> **Watch out:** cosine values in raw input embeddings are modest and noisy. Do not expect `king − man + woman = queen` to work perfectly in GPT-2's token table; that famous trick was shown on word2vec. Contextual embeddings (Part 6.2) are far richer.

**CHECKPOINT:** `king vs queen` is clearly higher than `king vs banana`.
*If not:* an `AssertionError ... splits into 2 tokens` means you changed a word — pick a common English word.

### SECTION 4.1.5 — Draw the embedding space

**WHERE:** same notebook
**GOAL:** Squash 768-dimensional vectors to 2-D and look at the clusters.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.1-E**:

{{code:4.1-E}}

**Step 3.** Press **Shift+Enter**.

> **You should see** a scatter plot like this one — royals in one corner, fruit and vehicles in another region, weekdays bunched together:

{{fig:4.1-E}}

**Line by line:** `V - V.mean(dim=0)` centres the cloud; `torch.linalg.svd` finds the two directions in which the 15 vectors spread out most (this is **PCA**); `V @ Vt[:2].T` projects every word onto those two directions so we can plot them.

**CHECKPOINT:** king/queen/prince are near each other and far from car/bus/truck.

**Common mistake at this level — confusing the tokenizer with the embedding.**
The tokenizer is fixed text-processing (string → IDs, no learning). The embedding is a trainable layer (ID → vector). *How to spot it:* if you think you can "train the tokenizer with backprop", or you load a model with a *different* tokenizer than it was trained with, outputs become garbage. Always load the tokenizer and model from the **same** name.

---

## 4.2 Topic 2 — The attention mechanism

**Concept.** Attention is a **soft lookup**. You have a *question* (query vector **q**), a set of *labels* (key vectors **K**) and the *content* behind each label (value vectors **V**). You score the question against every label with a dot product, turn the scores into percentages with softmax, and return the **weighted blend of the contents**. It was first used in 2014 to let a translation model look back at the most relevant source words [10]; the Transformer made it the *only* mechanism for mixing information between words [9].

![Attention as lookup](diagrams/d03_attention_lookup.svg)

### SECTION 4.2.1 — Do one attention step by hand

**WHERE:** paper and pen first, then the same notebook
**GOAL:** Calculate attention manually, then confirm with code.

The setup: three words, each with a 2-number key. Read number 1 as "is a noun" and number 2 as "is about sitting". So `cat = [1, 0]`, `sat = [0, 1]`, `mat = [1, 1]`. The query `q = [1, 0]` means "I am looking for a noun".

**Step 1.** On paper, compute the scores `q · key`:
cat: 1×1 + 0×0 = **1**; sat: 1×0 + 0×1 = **0**; mat: 1×1 + 0×1 = **1**.

**Step 2.** On paper, compute softmax of `[1, 0, 1]`:
e¹ = 2.718, e⁰ = 1, e¹ = 2.718; sum = 6.437 → weights **[0.422, 0.155, 0.422]**.

**Step 3.** On paper, blend the values `[10,0]`, `[0,10]`, `[5,5]`:
0.422×[10,0] + 0.155×[0,10] + 0.422×[5,5] = [4.22+2.11, 1.55+2.11] = **[6.33, 3.67]**.

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **4.2-A**:

{{code:4.2-A}}

**Step 6.** Press **Shift+Enter**.

> **You should see:**
{{out:4.2-A}}

**CHECKPOINT:** your paper answer matches `[6.335, 3.665]` (tiny rounding differences are fine).

### SECTION 4.2.2 — Why we divide by √d_k

**WHERE:** same notebook
**GOAL:** See softmax "saturate" when vectors are long, and how scaling fixes it.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.2-B**:

{{code:4.2-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.2-B}}

**Meaning:** a dot product of two random vectors with `d` numbers has a spread (standard deviation) of about √d. At d = 512 the scores are so large that softmax gives **one word 100% and all others 0%** — the gradient for the others becomes ~0 and learning stalls. Dividing by √d brings the spread back to ≈1 so softmax stays "soft". That is the `/ √d_k` in the formula.

**CHECKPOINT:** at `d=512`, unscaled biggest weight is `1.000` and scaled is much smaller.

### SECTION 4.2.3 — Write the general attention function

**WHERE:** same notebook
**GOAL:** One reusable function `attention(Q, K, V, mask)` that you will use for the rest of Part 4, verified against PyTorch.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.2-C**:

{{code:4.2-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.2-C}}

**Line by line**

- `K.transpose(-2, -1)` flips the last two axes so `Q @ Kᵀ` compares every query with every key → a `T × T` score grid.
- `masked_fill(mask == 0, -inf)` — where the mask says "not allowed", put −∞. `e^(−∞) = 0`, so those positions get exactly 0 weight. You will use this in Topic 5.
- `torch.softmax(scores, dim=-1)` normalises **each row** (each query) to sum to 1.
- `F.scaled_dot_product_attention` is PyTorch's official, fast (fused, FlashAttention-style) version [34]. Matching it proves your maths is right.

**CHECKPOINT:** both checks print `True`.

**Common mistake at this level — softmax over the wrong axis.**
`softmax(scores, dim=0)` normalises over queries instead of keys. The code still runs but the model learns nonsense. *How to spot it:* `w.sum(-1)` is not all ones. Always use `dim=-1` (last axis = keys).

---

## 4.3 Topic 3 — Self-attention

**Concept.** In **self**-attention the queries, keys and values all come from the **same sentence**. Every token's embedding `x` is multiplied by three learned matrices: `Q = x·W_Q`, `K = x·W_K`, `V = x·W_V`. Then every token attends to every token of its own sentence. The output is a new vector per token that has "absorbed" context — "it" can pull in information from "animal". Use it whenever tokens in one sequence need to understand each other (every Transformer layer does this).

![Self-attention](diagrams/d04_self_attention.svg)

### SECTION 4.3.1 — Build a self-attention layer

**WHERE:** same notebook
**GOAL:** Wrap `attention()` in an `nn.Module` with learnable `W_Q, W_K, W_V` and run it on our 6-word sentence.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.3-A**:

{{code:4.3-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.3-A}}

**Line by line**

- `nn.Linear(d_model, d_k, bias=False)` is a learnable matrix of size `d_model × d_k`. Three of them make Q, K, V from the same input.
- `X.detach().unsqueeze(0)` — `detach()` cuts the link to the embedding's gradient (we only want the numbers); `unsqueeze(0)` adds a batch axis: `(6, 4)` → `(1, 6, 4)`.
- The weight grid is 6 × 6: row = word looking, column = word looked at. Each row sums to 1.
- Rows 0 and 4 are identical — both are "the", with the same embedding and **no position information**. Hold that thought for Topic 6.
- The weights are fairly flat because `W_Q, W_K, W_V` are random. Training is what makes them sharp and meaningful — next you look at a *trained* model.

**CHECKPOINT:** output shape `(1, 6, 4)`, weights shape `(1, 6, 6)`.

### SECTION 4.3.2 — Find a real "it → animal" head in GPT-2

**WHERE:** same notebook (uses `gpt2` loaded in 4.1-D)
**GOAL:** See trained self-attention resolve a pronoun.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.3-B**:

{{code:4.3-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:4.3-B}}

**Line by line**

- `output_attentions=True` makes the model return the attention weights of all **12 layers × 12 heads**.
- `torch.stack(res.attentions)[:, 0]` → shape `(layers, heads, T, T)`; `[:, 0]` drops the batch axis.
- `"Ġit"` — GPT-2 writes a leading space as `Ġ`.
- `att[:, :, it_pos, animal_pos]` = for every layer and head, how much "it" attends to "animal". `argmax` finds the strongest head; `divmod` converts the flat index back to (layer, head).
- One head puts **85%** of "it"'s attention on "animal" and **0%** on "street". Nobody wrote a pronoun rule; training discovered it.

**CHECKPOINT:** the best head puts a clearly bigger weight on "animal" than on "street".

### SECTION 4.3.3 — Plot the attention map

**WHERE:** same notebook
**GOAL:** A heatmap of that head.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **4.3-C**:

{{code:4.3-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**

{{fig:4.3-C}}

**How to read it:** each row is a word "asking"; bright squares show where it looks. The upper-right triangle is black because GPT-2 is a *decoder* and cannot look at future words (Topic 5). The bright square in row **it**, column **animal** is the coreference you just measured. The bright first column is a known "attention sink": heads park unused attention on the first token.

**CHECKPOINT:** row "it" is brightest under "animal". Save your notebook: **File > Save** (Ctrl+S).

**Common mistake at this level — believing one attention map "explains" the model.**
A single head is one of 144 in GPT-2 small, and information also flows through MLPs and residuals. *How to spot it:* if you are making claims from one heatmap, check other layers/heads and test with interventions (Part 7.3 shows a proper measurement).

**Level-up check (Part 4 done):** you can explain "embedding = learned lookup table", compute attention by hand, and you have seen a trained head resolve "it".
