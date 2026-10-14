# LAB MANUAL — PART 11: Practice Projects

No solutions — only the goal, the requirements and how you will know you succeeded.

## Project 1 — Easy: FAQ assistant for one product

**Goal:** fine-tune a small open instruct model (1-3B) to answer questions about one product's manual.
**Requirements:** 300+ training examples built with `peftkit.data.build`; a 50-question human-written test set; LoRA on one rented GPU; baseline and tuned evaluation.
**Success criteria:** compare verdict PASS; at least one regression (an item that got worse) found and explained in writing; total GPU cost recorded.

## Project 2 — Medium: structured extraction with a public API

**Goal:** fine-tune a 7-8B model with QLoRA to turn free-text maintenance reports into strict JSON, and serve it to two "customers".
**Requirements:** a `must_match` JSON check in the test set; general and safety suites; merged model on the Hub; vLLM serving; two API keys with different rate limits; load test with a stated latency SLO.
**Success criteria:** JSON validity rate above your target on held-out reports; parity PASS; release gate RELEASE; each customer's usage visible in `/admin/usage`.

## Project 3 — Hard: a non-LLM architecture, end to end

**Goal:** adapt a model that is not a chat LLM (an encoder classifier, an ASR model, or a model you write yourself) using `uadapt`, and serve it.
**Requirements:** `UniversalPipeline.plan()` output saved; the derived loss or your own with a written justification; smoke test READY; task-appropriate metrics from the registry (e.g. macro-F1, WER, Recall@K) before and after; merge proof; an API in front; capacity and monthly cost computed.
**Success criteria:** a stranger can reproduce your numbers from your repository and README without asking you a question.
