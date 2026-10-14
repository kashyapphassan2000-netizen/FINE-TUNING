# 0. How to use this book

Type: **CONCEPT guide** with runnable labs. Five topics, one question underneath all of them:

> After pretraining, how do you make a model do **your** task, in **your** format, reliably?

That is supervised fine-tuning. Chapters 1-5 are the five topics from the syllabus: SFT as stage 1, instruction tuning (FLAN / Alpaca / OpenHermes), chat fine-tuning, chain-of-thought fine-tuning, and domain-specific practice. Chapters 6-9 are the exercises, interview questions, checklist and cheat sheet.

## 0.1 Where this sits in the series

|Volume|Covers|
|---|---|
|1 Universal PEFT Build Guide|LoRA/QLoRA/DoRA/AdaLoRA/LoRA+, data pipeline, GPUs, merging, vLLM|
|2 Fine-Tuning From Zero|training mechanics, tokenization, masking, evaluation, debugging|
|3 KV Cache & Attention|serving memory, attention variants, positional encodings|
|4 Enterprise Synthetic Data|generating and validating training data|
|**5 this book**|**what SFT is, what the standard datasets taught us, and how to do it well**|

## 0.2 What was executed

:::honest Every number here came from a real run in this container, 19 September 2026
- 5 new labs (50-54) run on CPU. They train a small randomly initialised Llama with the repository's own LoRA trainer and measure what changed.
- Results are scored by **likelihood over fixed candidate answers**, not free generation. A 200k-parameter random model generates noise; scoring isolates what SFT actually changed.
- Uncomfortable results are reported as they came out: 6 examples produce perfect memorisation and zero generalisation (lab 50); template diversity helped on unseen phrasing but was noisy (lab 51); the CoT lab is explicitly labelled as a mechanics demo, not evidence about CoT quality.
- Dataset facts were checked on the web on 19 September 2026: FLAN 2022 = 1,836 tasks, Apache-2.0; Alpaca = 52K records generated with text-davinci-003 from 175 seeds, licensed **CC BY-NC 4.0** (non-commercial).
- NOT executed: anything on a real pretrained model (no GPU here). Every claim that needs one is labelled.
:::

## 0.3 What a "toy model" result can and cannot tell you

|Can|Cannot|
|---|---|
|show the mechanics (what is masked, what is trained, what the data shape does)|predict quality numbers for a 7B model|
|show failure modes (memorisation, no generalisation from narrow data)|settle "does CoT help?"|
|show relative effects with everything else held constant|transfer hyperparameters|

Where a claim needs a real model, this book says so instead of inventing a benchmark.
