# 0. How to use this book

:::tip Start with the LAB MANUAL at the front
If you are new, do the **Lab Manual (Parts 0-14)** first: it is hand-holding, one action per step, with real outputs and a checkpoint after every section. Come back to these reference chapters when the manual points you here.
:::

This is the **single merged edition** of the Universal PEFT Build Guide and Fine-Tuning From Zero, rebuilt as **one pipeline in execution order**, plus the post-training stages both books were missing: evaluating the *served* model, load testing it, and a release gate that must pass before you scale.

If you are a beginner, read chapter 19 (plain-English primer) first, then follow chapter 2 (the runbook) line by line. Every runbook step points to the chapter that explains it.

## 0.1 Label legend

|Label|Where|Shell|
|---|---|---|
|LAPTOP · PowerShell|your Windows laptop|PowerShell|
|LAPTOP · WSL bash|your laptop, inside WSL Ubuntu|bash (FakeCloud and the `.sh` scripts)|
|GPU VM · bash|the rented Hyperstack VM, through VS Code Remote-SSH|bash|
|LAPTOP or GPU VM · bash|either machine|bash|
|BROWSER|a web page|clicks|
|FILE path|a file in the repo — copy it from the ZIP, not from this PDF|-|
|EXPECTED OUTPUT / REAL OUTPUT|what you should see / what was actually printed when this book was built|-|

## 0.2 Book structure = pipeline order

|Part|Chapters|You finish it with|
|---|---|---|
|0 Start here|0-2|how to use the book, the honest gap list, the command runbook|
|**O THE FLOW**|**3-16**|**your 13 steps in working order: is it open → hardware → data from reality → data gate → rent+connect → stack → load → JSONL → run → evaluate+tune → publish → serve+scale**|
|Orientation|31-32|architecture and the requirement map|
|A Foundations|19-29|you know what training, tokens, masking and LoRA actually do|
|B Setup|30-32|laptop ready, `pytest` green, rehearsal passed|
|C Sizing and compatibility|33-37|you know whether your model fits, on what, and for how much|
|D Data|38-51|a clean, measured, leak-free dataset|
|E Train|52-57|a trained adapter and a best checkpoint|
|F Evaluate|58-62|a measured before/after verdict, forgetting checked|
|G Beyond SFT|63-64|optional CPT and DPO|
|H Ship and scale|65-72|merged, served, **evaluated as served**, **load tested**, **gated**, autoscaled|
|I Finish|73-79|project, capstone, deliverables, troubleshooting, the brief this book follows|
|J Any architecture|81-104|plain-English primer first, then the model-agnostic method: identify, classify, trace, objective, targets, custom LoRA, every family, debugging, evaluation, serving stack|
|K Hardware, cost, public API|105-113|providers, GPU maths for training and serving, loading maths, cost, API keys for anyone, autoscaling with users, free hosting, showcase|
|L Mastery|114-117|
|N Universal algorithm as code|130-135|processor layer, first-principles derivation (worked on BERT and four others), loss derivation rules, the modality-independent pipeline, and a modality invented for the book, trained end to end|
|M Task abstraction and scope|118-129|the external review answered, the 5 support levels, modality × architecture × task × method, the task registry with 19 tasks and their metrics, classification / embeddings / rerankers / reward models, diffusion, TTS and video, multimodal depth, the corrected scope claim, lineage and licences, the engine, the harder challenge|the one-shot template, the unknown-model challenge, the decision tree, answer keys|

## 0.3 What was actually executed for this edition

:::honest Verification status, 19 September 2026 (CPU container, no GPU)
- **Full-pipeline rehearsal executed** (`scripts/rehearsal_cpu.py`): data → doctor → baseline eval → LoRA training → eval → compare → merge (max logit diff 6e-7) → serve behind an OpenAI API → API-key gateway (401 without a key) → evaluate the SERVED model → offline/served parity (gap 0.0) → load test ramp → release gate **RELEASE** → autoscaling settings derived from the load test. Output is reproduced in chapter 2.
- Repository test suite: **101 passed, 1 skipped** (the skip is the QLoRA test, which needs an NVIDIA GPU).
- All concept labs of both source books were executed; every REAL OUTPUT box is copied from a run.
- Checked against source, NOT executed: QLoRA on a GPU, vLLM 0.29.0 serving, multi-GPU accelerate configs, Hyperstack API calls, Kubernetes/KEDA manifests, distilabel/DataDreamer/Argilla examples.
- Not verified at all: the `vllm/vllm-openai:v0.29.0` Docker tag, the Hyperstack DELETE-VM path, real GPU memory and latency, any price.
- Library versions (PyPI, checked 17-19 Sep 2026): torch 2.14.0, transformers 5.17.0, peft 0.21.0, trl 1.13.0, accelerate 1.15.0, datasets 5.0.1, bitsandbytes 0.50.2, vllm 0.29.0 (pins torch 2.13.0 — separate environment).
:::
