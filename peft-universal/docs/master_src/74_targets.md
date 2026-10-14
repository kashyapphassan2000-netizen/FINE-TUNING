# 74. Step 5 — Discover adaptation targets by computation, not by name

:::why In one line
Names are conventions (`q_proj`, `query`, `q`, `in_proj`, `foo_7`). Computation is not. Find the attention projections because they receive the same input; find the MLP because it expands then contracts; find the head because it runs last or matches the vocabulary.
:::

## 74.1 The rules

@file src/uadapt/targets.py

|Evidence (from one traced forward pass)|Role|Adapt?|
|---|---|---|
|≥3 linears receive the SAME input tensor|attention q/k/v|yes|
|the first linear after that group|attention output projection|yes|
|2 expanding linears receive the same input|gated MLP (gate, up)|yes|
|one linear with out ≥ 2.5×in|fused qkv, or MLP expansion|yes|
|out ≥ 1.5×in / in ≥ 1.5×out|MLP up / down|yes|
|last linear executed, or out = vocab / num_labels|head|train fully or freeze — not LoRA by default|
|out = num_experts|MoE router|never (routing stability)|
|narrow (< width/8), e.g. an SSM dt projection|sensitive small projection|skip by default|
|Embedding, *Norm|embeddings, normalisation|no LoRA (norms are tiny: train fully if at all)|
|Conv1d / Conv2d (groups=1)|convolution|conv-LoRA (chapter 75)|

## 74.2 One methodology, twelve families

@file labs/lab60_universal_matrix.py
@lab lab60_universal_matrix

Read the second table: BERT's `query/key/value`, T5's `q/k/v/o/wi/wo`, ViT's `q_proj/k_proj/v_proj`, wav2vec2's convolutional feature extractor and Mamba's `in_proj/out_proj` were all found **without a name list**. The role labels are heuristics — for T5's cross-attention, q reads the decoder state while k/v read the encoder output, so the labels differ from self-attention — but the target **set** is right, and every family trained with an exact merge.

## 74.3 When names are unfamiliar, reason like this

1. Trace one batch (chapter 71). Group linears by the input tensor they receive.
2. Read the in/out widths: square, expanding, contracting, or ending at the vocabulary.
3. Read the order: what runs just before and just after.
4. Only then look at the names — to confirm, not to decide.
5. Verify: after injection, one backward must give every adapter a gradient.

:::check
- My target list came from a trace, not from a blog post's `target_modules`.
- Heads, routers, embeddings and norms are deliberately excluded or deliberately trained.
:::
