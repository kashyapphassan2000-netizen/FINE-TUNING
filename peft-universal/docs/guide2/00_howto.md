# 0. How to use Volume 2

Volume 1 (*Universal PEFT Build Guide*) teaches the five PEFT methods and how to ship them. It assumed you already understood training, tokenization and evaluation. **This volume removes that assumption.** It is the fundamentals, measurement and debugging book.

The test of this volume is not "can you recite what LoRA is". It is:

> Take a causal LM you have never seen, inspect it, build your own dataset, choose a method, train it, debug it when it fails, evaluate it honestly against the baseline, improve it, merge it and deploy it — on your own.

Chapter 20 is that exact challenge, with a grading rubric.

## 0.1 What is in here

|Part|Chapters|What it fixes|
|---|---|---|
|A. Fundamentals|1-4|why training works at all: forward, loss, backprop, optimizer; transformer internals; tokens; labels and masking|
|B. Data|5-6|the full dataset pipeline and how to measure dataset quality (including leakage)|
|C. Method|7-8|hyperparameters with reasons, and experiment design that lets you compare runs|
|D. Truth|9-11|evaluation, catastrophic forgetting, overfitting vs underfitting|
|E. Reality|12-16|model compatibility, debugging labs, method comparison, failure analysis, RAG vs fine-tuning|
|F. Beyond SFT|17-18|continued pretraining and preference optimization (DPO)|
|G. Proof|19-21|the end-to-end project, the unseen-model capstone, and the master checklist|

## 0.2 What was executed for this volume

:::honest Every number and every error message in this volume came from a real run
- 13 new labs (11-23) were executed on CPU in this container. Their outputs are pasted verbatim into the chapters, including one lab whose result was **negative** and is reported as such (chapter 16).
- New code, all tested: `compat.py` (can this model be fine-tuned here?), `diagnostics.py` (pre-flight doctor), `evaluate.py` (baseline vs tuned, regressions, forgetting), `experiments.py` (grid runner and results table), `tokenize_inspect.py` (token-by-token view).
- Checked against the real libraries: transformer internals in chapter 2 are compared to `transformers` 5.17's own RMSNorm, RoPE, SDPA attention, SwiGLU MLP and KV cache, and all five match.
- NOT executed: anything needing a GPU (QLoRA rows, VRAM figures) or the internet (real model downloads). Those are labelled where they appear.
- The whole repository test suite is run at the end of chapter 21.
:::

## 0.3 How to read it

Beginners: chapters 1 → 4 in order, run each lab yourself, then 5 → 11, then the project. Practitioners who already ship models: 8, 9, 12, 13, 15 are the ones that usually raise the quality bar.

Labels are the same as Volume 1: `LAPTOP · PowerShell`, `GPU VM · bash`, `FILE path`, `EXPECTED OUTPUT`, `REAL OUTPUT` (executed here). Copy code from the ZIP.
