# 0. How to use this book

Type: **CONCEPT guide**. Thirteen topics, one thread running through all of them:

> A transformer that generates text must remember the past. How it remembers decides how much memory you burn, how many users you can serve, and how long a context you can support.

Everything here — KV cache, FlashAttention, SDPA, MHA/MQA/GQA/MLA, PagedAttention, positional encodings, RoPE — is an answer to that one problem.

## 0.1 Structure of every chapter

Intuition → the problem it solves → the maths with a worked number → code from scratch → the same thing with the standard library → common mistakes → where it appears in production. Chapters 15-17 are 10 exercises with solutions, 15 interview questions with answers, and a one-page cheat sheet.

## 0.2 What was executed

:::honest Every number and output in this book came from a real run on CPU, 18 September 2026
- 8 new labs (30-37) were run in this container. Highlights, all reproducible with `python labs/labNN_*.py`:
  - the KV-cache formula predicted **4,096 bytes** and PyTorch allocated **4,096 bytes** (lab 31);
  - a from-scratch streaming ("flash") softmax matched standard attention to **2e-07** (lab 32);
  - a mini paged allocator wasted **1.9%** of KV memory where contiguous reservation wasted **81%** (lab 35);
  - MHA/GQA/MQA/MLA implemented and run: **4,096 / 1,024 / 256 / 288** KV bytes per token for the same model width (lab 34);
  - RoPE's relative-distance property verified numerically against `transformers` (lab 36).
- Library facts were read from installed source or official repositories, not memory: `torch` 2.14.0 SDPA signature and backends, `transformers` 5.17.0 attention implementations and RoPE types, vLLM v0.29.0 cache configuration.
- NOT executed: anything needing a GPU. FlashAttention kernels, fp8 KV cache, real throughput and any GPU timing are described from source and documentation and are labelled where they appear. No benchmark number is invented.
- New tool shipped with this book: `peftkit.kv`, a KV-cache calculator for any model.
:::

## 0.3 Versions used

|Package|Version|Checked|
|---|---|---|
|torch|2.14.0|installed here|
|transformers|5.17.0|installed here|
|vllm|0.29.0|source read at tag v0.29.0|
|flash-attn|2.8.3.post1|PyPI, 18 Sep 2026|

## 0.4 Prerequisites

You should know what a matrix multiply is and roughly what attention does. Volume 2, chapter 2 covers the transformer block if you need it first. Nothing else is assumed.
