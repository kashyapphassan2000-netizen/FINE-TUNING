# 31. Deliverables, README, GitHub, resume

## 31.1 Deliverables checklist

|Artifact|Where|
|---|---|
|code + configs + tests|GitHub (private or public)|
|dataset card + quality report|`reports/quality.json`, dataset repo on the Hub (if shareable)|
|adapter + merged model + model card|Hugging Face Hub|
|run evidence|`outputs/<run>/run_meta.json` (versions, loss curve, peak VRAM, eval)|
|serving proof|screenshot or log of `/health` and one completion; gateway metrics|
|this guide|PDF + ZIP|

## 31.2 README

@file README.md

## 31.3 Model card essentials (put in the Hub README)

Base model and licence, method and hyperparameters (copy from `run_meta.json`), data sources and synthetic share, filters applied, eval method and results you measured, intended use and known limits, how to serve (merged vs adapter).

## 31.4 Resume bullets (fill the brackets with YOUR measured numbers)

- Built a config-driven PEFT platform (LoRA, QLoRA, DoRA, AdaLoRA, LoRA+) on PyTorch/PEFT/TRL that fine-tunes any Hugging Face causal LM from [X]M to [Y]B parameters with one command; [N] automated tests.
- Designed a synthetic-data pipeline (grounded QA, Self-Instruct, LLM-as-judge, MinHash dedup, PII scrubbing, contamination and poisoning scans) that produced [N] training examples from [M] documents at [$Z] API cost.
- Fine-tuned [MODEL] with [METHOD] on [GPU], reaching [METRIC = value] vs [baseline value] on a human-verified eval set; peak VRAM [G] GB.
- Deployed merged and multi-LoRA models with vLLM behind an API-key gateway with rate limiting, failover and queue-based autoscaling; served [QPS] at p95 latency [L] ms.
- Diagnosed and fixed silent failures: AdaLoRA rank allocation never triggered by HF Trainer, chat-template token counting broken in transformers 5, CUDA driver/wheel mismatch on cloud GPU images.
