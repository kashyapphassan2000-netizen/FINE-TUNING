# 126. The corrected scope claim

:::why The review's point 19, accepted
The old framing — "fine-tune ANY model" — was broader than what is implemented. Here is the claim this book can defend, line by line.
:::

## 126.1 What this book is

> **Universal Fine-Tuning Engineering Guide** — an end-to-end, model-agnostic **methodology** for adapting AI models, with a **complete, tested implementation for decoder-only language models**, a **tested model-agnostic toolkit** (`uadapt`) that has been exercised on 12 architecture families and 10 training tasks, and **honest boundaries** where implementation stops and engineering begins.

## 126.2 What "universal" means here, precisely

|Claim|Status|
|---|---|
|the METHODOLOGY (identify → classify → trace → objective → targets → adapt → verify → train → evaluate → export → serve) applies to any trainable model|**yes** — that is what Parts J and M are|
|the target discovery, custom LoRA, gradient checks, training loop and exact merge work on any `nn.Module` with Linear/Conv layers|**yes, tested** on 12 families (lab 60) and 10 tasks (lab 62)|
|the task registry gives you the schema, loss, collator, metrics and serving note for 19 tasks|**yes**|
|a complete, ready-to-run pipeline exists|**only for decoder-only text SFT/DPO** (Parts A-I) — everything else needs your data code|
|it can fine-tune ANY model automatically, with no new code|**no, and it never will**: level 4-5 models need custom engineering (chapter 119)|
|TTS, video and production-scale diffusion are implemented|**no** — registered with recipes and metrics, not executed (chapters 123-124)|
|RLHF with PPO/GRPO|**no** — reward modelling is shipped, the RL loop is not|

## 126.3 The sentence to use

- Accurate: "a model-agnostic methodology plus a tested toolkit, with a complete pipeline for language models and documented extension points for vision, audio, multimodal, MoE, SSM, diffusion and custom architectures."
- Not accurate: "fine-tune any AI model automatically."

## 126.4 Why the honest version is stronger

Anyone who has fine-tuned a VLM or a TTS model knows that a single pipeline cannot cover them. A guide that says so, shows exactly where the boundary is, and hands you the tools to cross it yourself is more useful — and more credible — than one that claims the boundary does not exist.
