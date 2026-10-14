# 89. Evaluation for any model: build it, benchmark it, prove it

## 89.1 The shape is universal

```text
baseline (untuned model)  -> same suites ->  scores
fine-tuned model          -> same suites ->  scores
compare -> deltas + regressions + verdict -> failure analysis -> fix -> repeat
suites: DOMAIN (your task) · GENERAL (what it must not forget) · SAFETY · REGRESSION (past failures) · HUMAN sample
```

`peftkit.evaluate` implements this for text generation (chapter 44). For other families, the loop is the same; the metric changes.

## 89.2 Metrics by family

|Family / task|Primary metric|Also report|
|---|---|---|
|chat / instruction|task score on your suite (exact/contains/format), judge pairwise vs baseline|general + safety suites, human wins|
|classification|per-class recall, macro-F1|confusion matrix, calibration|
|extraction / QA|exact match, F1 over spans|refusal on unanswerable|
|summarisation / translation|human review + reference metrics|length, hallucination rate|
|ASR|WER / CER|per condition (noise, accent)|
|vision classification|accuracy, per-class recall|per-segment (lighting, device)|
|detection / segmentation|mAP@IoU / mIoU|per class|
|retrieval / embeddings|recall@k, MRR|per query type|
|diffusion|human preference, prompt adherence|FID-style distribution metric|
|any model|latency, throughput, cost per request|after serving (ch 53-54)|

## 89.3 Building your own evaluation set

1. Collect 50-300 **real** items; write the expected output or labels by hand, or have domain experts do it.
2. Hold out **entities** and **phrasings**, not just rows (Supervised Fine-Tuning book, chapter 5 idea; chapter 31 here).
3. Add general items (things it must still do) and safety items (things it must refuse).
4. Freeze the set, version it, hash it; pass it as `--contamination` to the data pipeline so it never leaks into training.
5. Add every fixed production failure as a regression item.

## 89.4 Public benchmarks with lm-evaluation-harness (facts read from lm-eval 0.4.13 source)

@file scripts/run_benchmarks.sh

- CLI flags used: `--model`, `--model_args`, `--tasks`, `--batch_size`, `--apply_chat_template`, `--output_path`, `--log_samples`, `--limit`, `--num_fewshot` (all present in the 0.4.13 source).
- Backends include `vllm`, `hf`, `local-completions`, `local-chat-completions`.
- **The chat-completions backend cannot run loglikelihood tasks** — its source raises `NotImplementedError("Loglikelihood is not supported for chat completions...")`. MMLU, HellaSwag, ARC and Winogrande are loglikelihood tasks, so run them with the `vllm` or `hf` backend on the merged model. Through your deployed API you can run generation tasks such as GSM8K and IFEval.
- `local-chat-completions` reads the key from `OPENAI_API_KEY`, and accepts `base_url`, `num_concurrent`, `max_retries` (verified in source).

:::warn Benchmarks are not your product
A general benchmark shows forgetting and broad capability. It does not show whether your model does YOUR task. Report both, and lead with your own suite.
:::

:::honest
lm-eval was not executed here: its tasks download datasets from the Hub, which this build machine cannot reach, and the useful runs need a GPU. The flags and backend behaviour above were read from its source.
:::

## 89.5 Showing the result to owners, customers or the public

Chapter 99: `peftkit.showcase` builds one HTML report from the files these tools produced — nothing typed by hand.
