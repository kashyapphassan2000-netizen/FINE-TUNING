# 70. Step 1 — Classify the architecture

:::why In one line
Family decides the objective, the data format, the trainer and the serving engine. Classify from **evidence**, and write the evidence down.
:::

## 70.1 The decision tree

```text
What does forward() take?
├── input_ids only ............................................ TEXT
│     ├── has decoder / decoder_input_ids / is_encoder_decoder -> ENCODER-DECODER (T5, BART)
│     ├── *ForCausalLM, causal mask .......................... -> DECODER-ONLY (Llama, Qwen)
│     └── *ForSequenceClassification / MaskedLM, no decoder .. -> ENCODER-ONLY (BERT, DeBERTa)
├── pixel_values ............................................... VISION (CNN, ViT, Swin)
│     └── + input_ids ....................................... -> VISION-LANGUAGE (multimodal)
├── input_values (waveform) / input_features (spectrogram) ..... AUDIO
│     ├── CTC head, lm_head over characters .................. -> CTC (wav2vec2, HuBERT)
│     └── decoder_input_ids ................................. -> SEQ2SEQ ASR (Whisper)
├── noisy sample + timestep .................................... DIFFUSION (U-Net, DiT)
├── anything, but:
│     ├── experts / router / num_experts in config .......... -> + MoE
│     ├── state_size, selective-scan projections ............ -> + STATE-SPACE (Mamba)
│     ├── LSTM / GRU / RNN modules .......................... -> + RECURRENT
│     └── attention AND SSM/recurrent layers ................ -> + HYBRID
└── plain nn.Module, no config, no Trainer ..................... CUSTOM
```

Labels combine: Whisper is `audio + encoder-decoder + text`; Mamba is `decoder-only + state-space + text`.

## 70.2 Every branch: what, why, how to detect, what changes

|Family|How to detect (evidence)|What changes in fine-tuning|
|---|---|---|
|decoder-only|`*ForCausalLM`, causal attention mask|next-token CE, chat template, assistant-only masking (chapters 27-29), `peftkit.train`|
|encoder-only|classification/MLM head, no decoder, bidirectional attention|task head trained from scratch (often fully), CE on class labels or MLM tokens, no chat template, no generation|
|encoder-decoder|`is_encoder_decoder`, `decoder_input_ids`, cross-attention|labels are target tokens; the model shifts them into decoder inputs (teacher forcing); cross-attention is a LoRA target|
|vision|`pixel_values`, Conv2d / patch embedding|image processor (resize, normalise), augmentation, head for the label space, conv-LoRA or attention LoRA|
|audio|`input_values` / `input_features`, sampling rate, mel bins|feature extractor at the model's sampling rate, CTC or seq2seq labels, length/frames constraints|
|multimodal|separate `vision_config` / `audio_config` / `text_config`|which component to adapt (encoder, projector, language model); processor handles both modalities|
|MoE|`num_experts`, router linears, experts|target attention first; routers frozen; fused experts not quantizable by bitsandbytes (chapter 22)|
|state-space|`state_size`, depthwise Conv1d + `in_proj` / `x_proj` / `dt_proj`|adapt the big projections (`in_proj`, `out_proj`); leave narrow dt projections|
|recurrent|`nn.LSTM` / `nn.GRU`|LSTM weights are packed parameters, not Linear: adapt surrounding Linears, or unfreeze the cell (FRAMEWORK LIMITATION for LoRA on the cell itself)|
|diffusion|forward takes `timestep`; noise-prediction output shape = input shape|MSE on predicted noise; LoRA on attention/linear layers of the denoiser; evaluate by sampling|
|custom|no Hugging Face config|you write the loss, the loop, the export (chapter 84)|

## 70.3 Automated, with evidence

`uadapt.inspect_model.classify` returns labels and the evidence behind each. In lab 60 it produced, for example: Whisper → `audio, encoder-decoder, text`; Mamba → `decoder-only, state-space, text`; the custom VLM → `custom, text, vision`.

:::warn Classification is evidence, not proof
Heuristics read configs and module types. A custom model can hide everything. When the labels look wrong, read `modeling_x.py`; when two labels conflict, trust the forward() source.
:::
