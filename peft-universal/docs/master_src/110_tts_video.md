# 110. TTS and video: registered, with the honest gaps

## 110.1 Text to speech

```text
text ─→ normalisation (numbers, dates, abbreviations) ─→ [phonemes] ─→ text encoder
     ─→ acoustic model (mel frames) or codec LM (audio tokens)  [+ speaker embedding]
     ─→ vocoder / codec decoder ─→ waveform
```

|Decision|Options|Consequence|
|---|---|---|
|representation|mel spectrogram vs neural audio codec tokens|mel → L1/L2 loss; codec tokens → cross-entropy (a language-model objective over audio tokens)|
|speaker|single-speaker fine-tune vs speaker embedding vs voice cloning from a reference clip|data volume and consent requirements differ completely|
|what to adapt|acoustic model (LoRA), speaker embedding only, vocoder (rarely)|vocoder retraining is a separate project|
|data|clean recordings, consistent microphone, aligned transcripts|noisy or misaligned data produces artefacts you cannot fix later|

**Evaluation**: human MOS-style listening tests (the real metric), speaker similarity, and the WER of an ASR model transcribing the output (intelligibility). Real-time factor for serving.

:::warn NOT SHIPPED
`TASKS["tts"]` registers the schema, the collator rules, the adaptation guidance and the metrics, but **no generic `loss_fn`**: TTS architectures differ too much for one honest default. You write the loss from the model's own training code. Nothing TTS was executed in this book.
:::

:::warn Consent and voice rights
Cloning a real person's voice needs their explicit, documented permission, and in many places is regulated. Record consent in the dataset lineage (chapter 113).
:::

## 110.2 Video

```text
video file ─→ decode ─→ frame sampling (uniform / stride / segment) ─→ clip [T, C, H, W]
          ─→ spatial encoder (+ temporal attention or 3D convolutions) ─→ head (class / caption / generation)
```

|Fact|Consequence|
|---|---|
|memory scales with **frames × resolution**|start with 8-16 frames at low resolution; grow only if the metric needs it|
|decoding is the bottleneck|pre-extract frames to disk, or you will train at 10% GPU utilisation|
|temporal labels|classification per clip vs per frame vs per segment changes the collator and the metric|
|adaptation|temporal layers and attention first; the spatial backbone is usually already good|

**Evaluation**: clip accuracy, per-class recall, temporal consistency; for captioning, the same human review as multimodal SFT. **Not executed here.**
