# 101. The unknown-model challenge

:::why How to use this chapter
Each challenge gives only what a real ticket gives you. Write down your answers to every question BEFORE reading chapter 103. The challenges get harder. If an answer in chapter 103 differs from yours, find the exact step where your reasoning diverged, correct it, then move on.
:::

For every challenge, determine: **architecture · inputs · outputs · loss · trainable layers · PEFT strategy · dataset format · trainer · memory · evaluation · deployment.** Use `uadapt.report`, a trace, and the chapters of Part J.

## Challenge 1 — a classifier nobody documented

- Files: `config.json` with `model_type: "deberta-v2"`, `architectures: ["DebertaV2ForSequenceClassification"]`, `num_labels: 7`, ~184M parameters.
- Data: 12,000 customer emails, each with one of 7 intent labels; the classes are imbalanced (one class 40%).
- GPU: one 24 GB card. Task: route incoming emails.

## Challenge 2 — speech in a factory

- Files: `config.json` with `model_type: "wav2vec2"`, `architectures: ["Wav2Vec2ForCTC"]`, `vocab_size: 32`; a `preprocessor_config.json` with `sampling_rate: 16000`.
- Data: 40 hours of operators reading maintenance codes, recorded at 44.1 kHz on phones, with transcripts.
- GPU: one 48 GB card. Task: transcribe codes; errors in digits are costly.

## Challenge 3 — a vision-language model for inspection photos

- Files: separate `vision_config`, `text_config`, a `projector` module, `processing_x.py` inserting `<image>` tokens.
- Data: 3,000 photos of welds with a one-paragraph inspector note each.
- GPU: two 80 GB cards. Task: generate the inspector note from the photo.

## Challenge 4 — a MoE you have never seen

- Files: `config.json` with `num_experts: 64`, `num_experts_per_tok: 6`, hidden 4096, 48 layers; `modeling_x.py` stores experts as 3-D parameters; `quantization_config: {"quant_method": "fp8"}`.
- Data: 30,000 legal Q&A pairs.
- GPU: 8 × 80 GB in one node. Task: domain chat assistant, served to 500 users.

## Challenge 5 — a custom model with no framework support

- Files: `model.py` defining `class ForecastNet(nn.Module)`: an input `nn.Linear(24, 128)`, a stack of `nn.GRU(128, 128, num_layers=2)`, and a `nn.Linear(128, 1)` head. No config, no Hugging Face class. Pretrained on energy demand from one country.
- Data: 18 months of hourly demand from a new country.
- GPU: none required (CPU). Task: next-hour demand for the new country.

## Challenge 6 — the hardest: an architecture that does not exist yet

- Files: `modeling_newarch.py` with a block that alternates a custom `MixerBlock` (two `nn.Linear` and a learned token-mixing matrix stored as an `nn.Parameter`) with standard attention built from `nn.MultiheadAttention`. Forward takes `input_ids`, returns logits; no loss.
- Data: 5,000 support conversations.
- GPU: one 48 GB card. Task: a chat assistant.
