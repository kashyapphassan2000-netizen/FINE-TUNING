# 67. PART J — What this guide establishes, and what it does not

:::why Why this chapter comes first
Chapters 5-65 were built and tested on **decoder-only causal language models**. Before generalising to any architecture, every concept must be sorted by how far it actually travels. Nothing here is assumed to "just work" elsewhere.
:::

## 67.1 The five-way classification

|Concept from chapters 5-65|Class|What it means for another architecture|
|---|---|---|
|forward → loss → backward → optimizer (ch 6)|1 architecture-independent|identical everywhere|
|LoRA maths W + (α/r)·B·A (ch 10), rsLoRA, LoRA+ learning-rate split (ch 14)|1 architecture-independent|applies to ANY linear map; conv needs the conv form (ch 75)|
|QLoRA's NF4 idea (ch 11)|4 needs modification|bitsandbytes quantizes `nn.Linear` only; convs, fused experts and custom layers stay unquantized (ch 22)|
|DoRA, AdaLoRA (ch 12-13)|3 framework-specific|as implemented, PEFT; the maths is general but you would re-implement it for custom layers|
|parameter counting on the meta device (ch 22)|1 architecture-independent|works for any class transformers (or your code) can build|
|trainable-parameter counting|1 architecture-independent|`sum(p.numel() for p in model.parameters() if p.requires_grad)`|
|data contracts, dedup, quality, leakage, synthetic data (ch 24-37)|1 architecture-independent|the checks are universal; the record format is not|
|chat templates, assistant-only masking, `-100` labels (ch 27-29)|2 decoder-only specific|encoders mask differently (MLM), seq2seq uses decoder labels, vision/audio have no chat template|
|`peftkit.train` (TRL SFTTrainer)|3 framework-specific|decoder-only text SFT; everything else uses `uadapt.loop` or the family's Trainer|
|baseline → tuned → compare, forgetting, failure analysis (ch 44-47)|1 architecture-independent|metrics change per task (ch 89)|
|KV cache, attention variants, RoPE|2 decoder-only specific|encoders have no KV cache at inference; SSMs keep a fixed state instead|
|vLLM serving, multi-LoRA, `serve_check` (ch 52, 57)|3 framework-specific|vLLM targets generative LMs and some multimodal models; do NOT assume it serves your architecture|
|served-model evaluation, load testing, release gate (ch 53-55)|1 architecture-independent|the idea is universal; the request format changes|
|gateway, API keys, autoscaling (ch 56, 95-96)|1 architecture-independent|any HTTP inference service|
|hardware planning (ch 21-23)|4 needs modification|the memory law holds; activation terms change for images/audio/video (ch 92)|
|MoE targeting (ch 22)|4 needs modification|depends on whether experts are Linear modules or fused tensors in that implementation|
|pre-quantized checkpoints (ch 22)|4 needs modification|each format (FP8, MXFP4, GPTQ, AWQ) has its own dequantization path|
|merging adapters into weights|5 completely changes for some|exact for linear and dense conv; NOT exact for grouped/depthwise convs, nonlinear adapters (IA3 is multiplicative), prompt/prefix tuning (nothing to merge)|

Classes: **1** architecture-independent · **2** decoder-only specific · **3** framework-specific · **4** needs modification for other modalities · **5** completely changes for another architecture.

## 67.2 Where the earlier chapters are silent

Whenever Part J needs something chapters 5-65 never covered, it says so with this marker:

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Chapters 5-65 do not establish this. The adaptation is derived from the model architecture and verified by a lab in Part J.
:::

Examples: vision processors, CTC labels, seq2seq decoder inputs, diffusion noise objectives, custom LoRA for convolutions, training models with no Hugging Face integration.

## 67.3 What Part J adds, and how it was verified

:::honest Executed for this part (CPU, 21 September 2026)
- New package `uadapt`: inspection, evidence-based classification, hook tracing, name-free target discovery, custom LoRA for Linear/Conv1d/Conv2d with exact merge, a generic training loop, strategy ranking, the `UniversalModelAdapter`, and a compatibility-report CLI. 8 tests.
- **Lab 60: one code path on 12 architecture families** — decoder-only (Llama), encoder-only (BERT), encoder-decoder (T5), vision (ViT and a custom CNN), audio CTC (wav2vec2), audio seq2seq (Whisper), state-space (Mamba), MoE (Qwen3-MoE), recurrent (custom LSTM), diffusion (custom denoiser), multimodal (custom vision + projector + LM). Every family trained, and every merge was exact.
- **Two real silent-failure bugs** were found by lab 60 and fixed in the code; they are taught in chapter 75.
- Lab 61: seven planted failures across architectures, each caught by a universal check.
- Public-API layer: self-serve keys, per-key limits and model access, metering (tested); capacity and cost maths (tested); stakeholder showcase report (tested); rehearsal extended to issue a key to a new user and serve them.
- All models in labs 60-61 are built from configs with random weights. They test the **method**, not quality on real data.
:::
