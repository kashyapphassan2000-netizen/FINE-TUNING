# 125. Multimodal in depth

Chapter 94 covered the anatomy and adapted a custom VLM. This chapter is the operational detail.

## 125.1 The pipeline, with the failure points marked

```text
raw record {image(s), conversation}
  → modality validation      ← corrupt files, wrong mode (CMYK/RGBA), 0-byte images, unsupported codecs
  → preprocessing            ← resize policy, aspect ratio, variable resolution, tiling for large images
  → processor                ← THE model's own processor: it inserts image placeholder tokens
  → conversation construction← where the image goes in the turn order
  → modality tokens          ← the number of placeholders MUST match the number of images
  → collator                 ← pad text, stack pixels, keep per-sample image counts
  → forward                  ← vision tower → projector → LM
  → selective loss           ← answer tokens only; image and prompt tokens = -100
```

## 125.2 The decisions that matter

|Decision|Options|When|
|---|---|---|
|what to adapt|projector only → + LM LoRA → + vision tower last blocks → full|in that order; stop when the metric stops moving|
|resolution|fixed square vs native aspect vs tiling|tiling raises tokens per image sharply: check the KV budget (chapter 36)|
|multiple images per sample|supported by some models only|placeholder counts and position embeddings must line up|
|missing modality|text-only samples in a multimodal dataset|either mask the vision path or drop the sample — decide, do not leave it to chance|
|processor version|pinned with the model revision|a processor change silently alters normalisation and token counts|

## 125.3 Evaluation

Task score on held-out **images** (not just held-out questions), field-level accuracy for structured outputs, hallucination rate (claims not visible in the image), refusal behaviour when the image does not contain the answer, and human review of a sample. Report per segment: lighting, device, subject type.

:::warn The two silent failures
1. **Placeholder mismatch**: the number of `<image>` tokens differs from the number of images — some models raise, others silently misalign.
2. **Loss on image tokens**: if the mask is wrong, the model trains to "predict" image placeholders and learns nothing useful. Verify by decoding the trained span of one sample (chapter 101).
:::
