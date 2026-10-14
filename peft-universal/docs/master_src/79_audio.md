# 79. Audio models (ASR, classification, speaker, TTS, audio-language)

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Derived from the architecture; exercised in lab 60 (wav2vec2 CTC, Whisper) and lab 61 (case 5).
:::

## 79.1 The flow

```text
waveform (float samples) at the model's SAMPLING RATE (commonly 16 kHz for speech models: check the config)
   -> FEATURE EXTRACTOR: raw normalised samples (wav2vec2: input_values) OR log-mel spectrogram (Whisper: input_features)
   -> AUDIO ENCODER (conv feature extractor downsamples time, then transformer blocks)
   -> CTC head (per-frame characters)  OR  decoder (seq2seq text)  OR  classification head
```

## 79.2 CTC vs seq2seq

|Aspect|CTC (wav2vec2, HuBERT)|Seq2seq (Whisper)|
|---|---|---|
|labels|token/character ids, no alignment needed|target token ids (teacher forcing)|
|constraint|**label length ≤ output frames** after downsampling|max target length|
|loss|CTC|CrossEntropy|
|decoding|greedy/beam over frames, collapse repeats and blanks|autoregressive generation|
|LoRA targets|attention q/k/v/out, feed-forward dense layers; conv extractor usually frozen|encoder + decoder attention, cross-attention, fc1/fc2|

Lab 61 case 5 is the classic CTC failure: 4,000 samples produce 399 output frames; a 449-token label gives `loss = inf`. `ctc_zero_infinity=True` hides the symptom; it does not fix the data.

```py
from transformers import AutoProcessor, AutoModelForCTC
proc = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
inputs = proc(audio_array, sampling_rate=16000, return_tensors="pt")          # must match the model's rate
inputs["labels"] = proc(text="THE PUMP LEAKS", return_tensors="pt").input_ids
loss = model(**inputs).loss
```

## 79.3 Evaluation and serving

- ASR: word error rate (WER) / character error rate on held-out speakers and conditions (noise, accents, microphones).
- Classification: accuracy, per-class recall; speaker: equal error rate.
- TTS: human listening tests (MOS-style) plus intelligibility via an ASR model.
- Serving: streaming vs batch changes the architecture of the service (chunked inference for long audio). vLLM support for audio models exists for specific architectures only — check before assuming (chapter 90).

:::warn Traps
Resampling mistakes (feeding 44.1 kHz audio to a 16 kHz model), silence padding counted as speech, and evaluating on the same speakers you trained on.
:::
