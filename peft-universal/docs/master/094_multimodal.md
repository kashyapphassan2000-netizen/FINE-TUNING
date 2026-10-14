# 94. Multimodal models (image + text, audio + text, video + text)

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Derived from the architecture; exercised in lab 60 (custom VLM row).
:::

## 94.1 The anatomy

```text
image / audio / video frames -> MODALITY ENCODER (ViT / audio encoder)   -> features
                              -> PROJECTOR (MLP / resampler)             -> "soft tokens" in the LM's width
text ------------------------> TOKEN EMBEDDINGS ----------------------\
                                                                      -> FUSION (concatenate / cross-attend)
                                                                      -> LANGUAGE DECODER -> LM head -> text
```

## 94.2 Where to adapt, in order of cost

|Component|Adapt when|Typical choice|
|---|---|---|
|projector|new domain imagery, alignment is off|train fully or LoRA (small)|
|language decoder|output behaviour/format must change|LoRA on attention + MLP|
|modality encoder|images/audio are unlike pretraining (medical scans, spectrograms of machines)|LoRA or unfreeze last blocks — most expensive|
|fusion / cross-attention|the model ignores the image|LoRA on cross-attention|

Labels: loss only on the text answer tokens; image/soft tokens and the prompt are masked with -100 (the same idea as chapter 42, applied to the visual prefix).

## 94.3 What lab 60 found on a custom VLM

The discovery adapted the vision convs, the projector and the language model's feed-forward layers; loss fell 4.057 → 1.701 on one batch, and the merge was exact **after** the effective-weight fix (chapter 89). The attention inside PyTorch's `nn.MultiheadAttention` packs q/k/v into one parameter, so it cannot be adapted by wrapping Linears: **FRAMEWORK LIMITATION** — re-implement the attention with separate Linear layers if you need it adapted (**CUSTOM ENGINEERING REQUIRED**).

## 94.4 Processor, collator, serving

- Use the model's processor for BOTH modalities; it also inserts the image placeholder tokens the model expects.
- Collators must pad text and stack pixel tensors, and keep the placeholder count consistent with the image count.
- Serving: vLLM supports specific vision-language architectures; otherwise transformers behind an API (chapter 104).
