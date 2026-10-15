# PART 6 — ADVANCED (Topics 7, 8, 9, 10)

You now wire blocks into the three families. You will (a) use real pretrained models of each family, (b) **train your own decoder-only GPT** from scratch, and (c) **build an encoder–decoder with cross-attention** from scratch and *watch* the cross-attention work.

![Three families](diagrams/d10_families.svg)

## SECTION 6.0 — Create the Part 6 notebook and paste the building blocks

**WHERE:** Google Colab
**GOAL:** Fresh GPU notebook containing the Part 5 classes.

**Step 1.** Click **File > New notebook in Drive**.

**Step 2.** Rename it to `T101_Part6_Advanced.ipynb`.

**Step 3.** Click **Runtime > Change runtime type**, choose **T4 GPU**, click **Save**. (Part 6 trains models — use the GPU.)

**Step 4.** Click **+ Code**.

**Step 5.** Paste cell **6.0-A**:

{{code:6.0-A}}

**Step 6.** Press **Shift+Enter**.

> **You should see:** `torch 2.x.x+cu12x | device: cuda`

**Step 7.** Click **+ Code**.

**Step 8.** Paste cell **6.0-B** (the exact classes you built and tested in Part 5 — a new notebook does not remember old notebooks):

{{code:6.0-B}}

**Step 9.** Press **Shift+Enter**.

> **You should see:**
{{out:6.0-B}}

**CHECKPOINT:** `device: cuda` and `building blocks ready`.
*If you see* `device: cpu`: redo Step 3. Everything still works on CPU, but Section 6.3 training will take ~30–60 minutes instead of a few.

---

## 6.1 Topic 7 — Encoder–decoder Transformers

**Concept.** The original 2017 Transformer [9] has two stacks. The **encoder** reads the whole input with *unmasked* self-attention and produces one context-rich vector per input token (the "memory"). The **decoder** writes the output one token at a time using (1) *masked* self-attention over what it has written so far, (2) **cross-attention** into the encoder's memory, (3) an MLP. Use it when the output is a *transformation* of a separate input: translation, summarisation, speech-to-text (Whisper [45]), T5 [13], BART.

![Encoder-decoder](diagrams/d09_encoder_decoder.svg)

### SECTION 6.1.1 — Use a real encoder–decoder (T5)

**WHERE:** Colab, `T101_Part6_Advanced.ipynb`
**GOAL:** Translate and summarise with T5-small (60 M parameters).

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.1-A**:

{{code:6.1-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.1-A}}

**Line by line**

- `AutoModelForSeq2SeqLM` loads an encoder–decoder model with a language-model head on the decoder.
- T5 was trained with a **task prefix** in the input text (`translate English to German:`, `summarize:`) — one model, many tasks.
- `t5.generate(...)` runs the encoder **once** on the input, then calls the decoder repeatedly, feeding back each new token, until it emits the end token or hits `max_new_tokens`.

**CHECKPOINT:** the German sentence is `Das Haus ist wunderbar.`

### SECTION 6.1.2 — Find the cross-attention inside T5

**WHERE:** same notebook
**GOAL:** See the three sub-layers of a real decoder layer.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.1-B**:

{{code:6.1-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.1-B}}

**Meaning:** every T5 decoder layer is exactly the diagram: `T5LayerSelfAttention` (masked) → `T5LayerCrossAttention` (looks at the encoder) → `T5LayerFF` (MLP). Encoder layers have only self-attention + FF. You will build this exact structure yourself in Section 6.4.

**CHECKPOINT:** you can point to `T5LayerCrossAttention` in the printout.

**Common mistake — feeding the wrong thing to an encoder–decoder.** Beginners pass the *target* text to the encoder, or forget the task prefix for T5. *How to spot it:* output repeats the input or is empty. Rule: input text → encoder; the decoder starts from its special start token and only sees what it has generated.

---

## 6.2 Topic 8 — Encoder-only Transformers

**Concept.** Keep only the encoder stack: no causal mask, so **every token sees every other token in both directions**. It cannot naturally write text, but it produces the best *understanding* vectors. It is trained by hiding ~15% of tokens and predicting them (**masked language modelling**, BERT [11]). Use it for classification, named-entity recognition, and — most commercially important — **embedding models for semantic search and RAG** (sentence-transformers, ModernBERT [44]).

### SECTION 6.2.1 — Fill in a masked word (BERT)

**WHERE:** same notebook
**GOAL:** Watch BERT use context on **both** sides of a gap.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.2-A**:

{{code:6.2-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see** (the model reads "barked" and "mailman" — words *after* the gap — to guess "dog"):
{{out:6.2-A}}

**Line by line:** `pipeline("fill-mask", ...)` wraps tokenizer + model + decoding. `[MASK]` is BERT's special hidden-word token. Each result has the guessed token and its probability. A decoder-only model reading left-to-right could not use "barked" when filling a gap that comes before it.

**CHECKPOINT:** `dog` is the top answer.

### SECTION 6.2.2 — Contextual embeddings: same word, different meaning

**WHERE:** same notebook
**GOAL:** Prove the vector for "bank" changes with its context (static embeddings from Topic 1 cannot do this).

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.2-B**:

{{code:6.2-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.2-B}}

**Line by line:** `last_hidden_state[0]` holds one 768-number vector per token *after* 12 layers of bidirectional self-attention. We pick the vector at the position of `bank`. The two "river" banks are more similar to each other than to the "money" bank — the encoder has used the surrounding words to separate the two meanings. This is the step from Topic 1's *static* embeddings to *contextual* embeddings.

**CHECKPOINT:** river-vs-river similarity is higher than river-vs-money.

### SECTION 6.2.3 — A 10-line semantic search engine (the core of RAG)

**WHERE:** same notebook
**GOAL:** Rank documents by meaning, not keywords.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.2-C**:

{{code:6.2-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.2-C}}

**Line by line**

- `all-MiniLM-L6-v2` is a small encoder-only model fine-tuned so that sentences with the same meaning get similar vectors.
- `padding=True` pads shorter sentences; `attention_mask` (1 = real token, 0 = PAD) is the **padding mask** from Topic 5.
- `(hidden * m).sum(1) / m.sum(1)` = **mean pooling**: average the token vectors, ignoring padding.
- `F.normalize` makes each vector length 1, so a dot product equals cosine similarity.
- The query shares **no keywords** with "How do I reset my password?" yet scores 0.68 vs ≤0.11 for the rest. That is how RAG systems find the right document chunk before an LLM answers.

**CHECKPOINT:** the password document is ranked first.

**Common mistake — using a decoder LLM's raw hidden states as search embeddings.** They are not trained for similarity and give poor retrieval. *How to spot it:* relevant documents rank randomly. Use a model trained for embeddings (sentence-transformers, BGE, E5, Qwen3-Embedding, etc.) and mean pooling or the model's documented pooling.

---

## 6.3 Topic 9 — Decoder-only Transformers

**Concept.** Keep only the decoder stack, **without** cross-attention: token embeddings + positions → N blocks of *masked* self-attention + MLP → a linear "LM head" that scores every vocabulary token as the next token. Training is trivially simple — predict the next token everywhere in huge amounts of text — and the same model can then do almost any task by continuing a prompt. GPT-2/3/4, Llama, Qwen, Mistral, DeepSeek and Claude-style models are all decoder-only. You will now build and train one.

### SECTION 6.3.1 — Load the data

**WHERE:** same notebook
**GOAL:** Download Tiny Shakespeare (1.1 MB of plays) and turn it into numbers.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.3-A**:

{{code:6.3-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.3-A}}

**Line by line:** we use a **character-level** tokenizer: every distinct character (65 of them — letters, punctuation, space, newline) is a token. `encode`/`decode` convert between text and IDs. 90% of the text trains the model; the last 10% is held out (**validation**) to check it is learning patterns, not memorising.

**CHECKPOINT:** vocabulary size `65`.

### SECTION 6.3.2 — Make training batches

**WHERE:** same notebook
**GOAL:** Random chunks of 128 characters, with targets shifted by one.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.3-B**:

{{code:6.3-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see** (your random text will differ):
{{out:6.3-B}}

**Meaning:** `y` is `x` moved one character left. At every position *t* the model sees characters 0…t and must predict character t+1. One 128-character chunk therefore gives **128 training examples at once** — this is why decoder-only training is so efficient, and it only works because of the causal mask.

**CHECKPOINT:** both shapes are `(64, 128)` and the `y` text equals the `x` text shifted by one.

### SECTION 6.3.3 — Build MiniGPT

**WHERE:** same notebook
**GOAL:** A complete decoder-only Transformer (~3.2 M parameters).

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.3-C**:

{{code:6.3-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.3-C}}

**Line by line**

- `tok_emb` = Topic 1; `pos_emb` = learned positional encoding (Topic 6, GPT-2 style).
- `self.blocks` = 4 × your Part 5 `Block` (Topics 3–5).
- `register_buffer("mask", ...)` stores the causal mask with the model so it moves to the GPU with `.to(device)`, but it is not a trainable parameter.
- `self.mask[:, :, :T, :T]` crops the mask when the input is shorter than 128.
- `ln_f` + `head`: final LayerNorm, then a Linear layer that outputs 65 scores (**logits**) per position.
- `F.cross_entropy` compares the logits with the true next characters. `view(-1, ...)` flattens batch × time into one long list.
- `generate`: repeatedly (1) crop to the last 128 tokens, (2) run the model, (3) take the **last** position's logits, (4) divide by `temperature` (lower = safer, higher = more random), (5) optional `top_k` keeps only the k likeliest characters, (6) sample one with `torch.multinomial`, (7) append and repeat.
- Before training, loss ≈ ln(65) = 4.17 — the model is guessing uniformly at random. Good sanity check.

**CHECKPOINT:** `MiniGPT parameters: 3,225,600` and initial loss near 4.2.

### SECTION 6.3.4 — Train it

**WHERE:** same notebook, T4 GPU
**GOAL:** Bring validation loss from ~4.2 down to ~1.5–1.7.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.3-D**:

{{code:6.3-D}}

**Step 3.** Press **Shift+Enter**.

**Step 4.** Wait. A line is printed every 300 steps.

> **You should see** (T4 GPU, about 3–6 minutes): `step 0` near 4.3, falling fast to ~1.9 by step 300, then slowly to roughly **train ≈ 1.3–1.4, val ≈ 1.5–1.6** at step 3000, followed by a plot of the two curves. Exact numbers vary run to run.
>
> For reference, here is the start of the same code on the CPU-only build machine used to test this guide (shortened to 600 steps, so it stops earlier):
{{out:6.3-D}}

{{fig:6.3-D}}

**Line by line**

- `estimate_loss` averages 50 batches for train and val, under `torch.no_grad()` (no gradients = faster) and `m.eval()` (turns dropout off).
- `AdamW` is the standard optimizer for Transformers. `lr=1e-3` is fine for a model this small; big LLMs use ~1e-4 to 3e-4 with warm-up (Part 7).
- `zero_grad → backward → clip_grad_norm_ → step` — the 5-line loop from Part 1, plus gradient clipping (caps very large updates; standard for Transformers).

| If you see... | It means... | Do this next |
|---|---|---|
| val keeps falling with train | Healthy learning | Continue |
| train ≪ val and val starts rising | Over-fitting (memorising) | Stop earlier, raise `dropout`, or use a smaller model |
| loss stuck at ~3.3 | Model only learned letter frequencies | Check the causal mask and that `y` is shifted by one |
| `CUDA out of memory` | Batch too big for the GPU | Set `BATCH_SIZE = 32`, re-run 6.3-B, 6.3-C and 6.3-D |

**CHECKPOINT:** final val loss below 1.8.

### SECTION 6.3.5 — Generate Shakespeare

**WHERE:** same notebook
**GOAL:** Sample 300 new characters from your model.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.3-E**:

{{code:6.3-E}}

**Step 3.** Press **Shift+Enter**.

> **You should see** Shakespeare-*shaped* text: speaker names in capitals followed by a colon, line breaks, mostly real English words, but nonsense meaning. Example from the shortened CPU test run (yours, trained 5× longer, will look better):
{{out:6.3-E}}

**CHECKPOINT:** the output contains `NAME:` lines and real words. You trained a GPT from scratch.

### SECTION 6.3.6 — Compare with a real 2026 decoder-only model (Qwen3-0.6B)

**WHERE:** same notebook (downloads ~1.5 GB)
**GOAL:** Read a modern decoder's configuration and chat with it.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.3-F**:

{{code:6.3-F}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.3-F}}

**Line by line**

- `dtype="auto"` loads the weights in the precision they were saved in (bfloat16) — half the memory of float32.
- The config shows **16 query heads but only 8 key/value heads** — that is **grouped-query attention (GQA)**, which you will build in Part 7 — and a 40,960-token context.
- `apply_chat_template` wraps your message in the special tokens the model was fine-tuned with; `enable_thinking=False` switches off Qwen3's long "thinking" mode for a short answer.
- `do_sample=False` = greedy decoding (always pick the most likely token) → repeatable output.
- `out[0][inputs["input_ids"].shape[1]:]` drops the prompt tokens so only the answer is printed.

**CHECKPOINT:** a one-sentence answer.

> **Watch out (brutally honest):** the answer above is *partly wrong* — decoder-only models are not "sequence-to-sequence" in the T5 sense; they predict the next token. A 0.6 B model writes fluent, confident text with factual errors. Never trust a small model's explanation of anything important; this is why companies evaluate models and use retrieval (RAG).

**Common mistake — forgetting the chat template.** Feeding raw text to an instruction-tuned model gives rambling or broken answers. *How to spot it:* the model continues your question instead of answering it. Always use `apply_chat_template` for chat models; use raw text only for base models.

---

## 6.4 Topic 10 — Cross-attention (and building an encoder–decoder from scratch)

**Concept.** **Cross-attention** is ordinary attention where the **queries come from one sequence** (the decoder's tokens) and the **keys and values come from another** (the encoder's output). It answers "for the word I am writing now, which input positions should I read?". It is how translation models align words, how Whisper's text decoder listens to audio, how Stable-Diffusion-style image models follow a text prompt, and how multimodal LLMs read image features. In code it is your `MultiHeadAttention` with `x_kv = encoder_output`.

![Cross-attention](diagrams/d11_cross_attention.svg)

The task: give the model 8 digits, it must output them **reversed**. To succeed, decoder step 1 must read source position 8, step 2 position 7, … — so a working model *must* learn an anti-diagonal cross-attention pattern. You will see it.

### SECTION 6.4.1 — Make the toy dataset

**WHERE:** same notebook
**GOAL:** Generate (source, decoder input, target) triples.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.4-A**:

{{code:6.4-A}}

**Step 3.** Press **Shift+Enter**.

> **You should see** (your digits will differ):
{{out:6.4-A}}

**Meaning:** tokens 0–9 are digits; token 10 is `BOS` ("begin sequence"). The decoder input is the target shifted right with BOS in front — **teacher forcing**: during training the decoder sees the *true* previous digits and learns to predict the next one.

**CHECKPOINT:** the target is the source reversed, and the decoder input is `[10]` + target without its last digit.

### SECTION 6.4.2 — Build the decoder block with cross-attention, and the full model

**WHERE:** same notebook
**GOAL:** An encoder–decoder Transformer (~0.2 M parameters) from your own parts.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.4-B**:

{{code:6.4-B}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.4-B}}

**Line by line**

- `DecoderBlock` has **three** sub-layers, exactly like T5 in 6.1.2: masked `self_attn`, then `cross_attn`, then `ffn`, each pre-normed with a residual.
- `self.cross_attn(self.ln2(x), x_kv=memory, ...)` — **this single line is cross-attention**: queries from the decoder state `x`, keys/values from the encoder output `memory`. It returns the weights so we can plot them.
- `encode()` runs the encoder blocks with **no mask** (bidirectional) and a final LayerNorm.
- `decode()` runs the decoder blocks with a causal mask; every block reads the same `memory`.
- Source and target get separate embedding tables but share a learned position table.

**CHECKPOINT:** `Seq2Seq parameters: 236,363`.

### SECTION 6.4.3 — Train it

**WHERE:** same notebook
**GOAL:** Loss from ~2.5 to almost exactly 0 in 1,500 steps (≈40 s on the CPU test machine, faster on T4).

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.4-C**:

{{code:6.4-C}}

**Step 3.** Press **Shift+Enter**.

> **You should see** (numbers vary slightly on GPU):
{{out:6.4-C}}

**CHECKPOINT:** the last loss is below 0.05.

### SECTION 6.4.4 — Decode new examples (no teacher forcing)

**WHERE:** same notebook
**GOAL:** Let the model write its own output, digit by digit, and measure accuracy.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.4-D**:

{{code:6.4-D}}

**Step 3.** Press **Shift+Enter**.

> **You should see:**
{{out:6.4-D}}

**Line by line:** `greedy_decode` encodes the source **once**, starts the decoder with `[BOS]`, and loops 8 times: run the decoder, take the most likely next digit (`argmax`), append it. This is exactly what `t5.generate` did in 6.1.1. Accuracy counts a sequence as correct only if all 8 digits are right.

**CHECKPOINT:** accuracy above 95%.

### SECTION 6.4.5 — Look at the cross-attention

**WHERE:** same notebook
**GOAL:** Visual proof that each output step reads the correct source position.

**Step 1.** Click **+ Code**.

**Step 2.** Paste cell **6.4-E**:

{{code:6.4-E}}

**Step 3.** Press **Shift+Enter**.

> **You should see:** a bright **anti-diagonal** — the first output digit looks at the last source digit, and so on:

{{fig:6.4-E}}

{{out:6.4-E}}

**Meaning:** nobody told the model "reverse"; it discovered that the right *alignment* is "step i reads position 7−i", and cross-attention is where that alignment lives. In translation you would see a mostly-diagonal pattern with jumps where word order differs between languages.

**CHECKPOINT:** the brightest cell in each row lies on the anti-diagonal. Click **File > Save**.

**Common mistake at this level — swapping Q and K/V in cross-attention.** If you take queries from the encoder and keys/values from the decoder, the output has the *source* length instead of the target length and the model cannot generate. *How to spot it:* a shape error when adding the residual (`x + c`), or an output of the wrong length. Rule: **Q from the side that is writing; K and V from the side being read.**

**Level-up check (Part 6 done):** you have used all three families, trained a GPT, and built cross-attention that you can see working.
