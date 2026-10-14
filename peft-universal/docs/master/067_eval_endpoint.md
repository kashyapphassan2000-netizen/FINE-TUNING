# 67. Evaluate the SERVED model

:::why In one line
The model you evaluated offline is not automatically the model your users get. Evaluate the endpoint with the same suites, and compare.
:::

## 67.1 What can change between training and serving

|Changes silently|Symptom|
|---|---|
|chat template (server default vs your training template)|format breaks, model rambles or repeats the prompt|
|dtype / quantization (bf16 → fp8/AWQ)|small score drop, occasionally large on arithmetic|
|`max_model_len` shorter than training|long prompts truncated, answers lose context|
|adapter name (multi-LoRA)|the base model answers instead of your adapter|
|sampling defaults (temperature, top_p)|non-deterministic scores|
|the gateway (timeouts, rate limits)|errors counted as bad answers|

## 67.2 The tool

@file src/peftkit/eval_endpoint.py

It sends each suite item to `/v1/chat/completions` at temperature 0, scores the answers with the same rules as the offline evaluator, records latency and errors per item, and writes a report in the same format — so the offline/served **parity** check and the before/after **compare** both work on it.

```gpu
python -m peftkit.eval_endpoint --url http://127.0.0.1:8000/v1 --model mymodel --api-key $VLLM_API_KEY \
  --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl \
  --out reports/served.json --offline reports/tuned.json --parity-out reports/parity.json
```

## 67.3 Rehearsed on the laptop first

@file src/peftkit/serve_local.py

`serve_local` is a small OpenAI-compatible server backed by transformers. It exists so the whole chain can run on a laptop; it exposes the same routes and the same `vllm:num_requests_waiting` metric name that vLLM does. In the rehearsal (chapter 2), the served model scored exactly like the offline one: **parity gap 0.0, zero errors**.

:::warn What parity does not prove
Parity means "serving did not change the model". It does not mean the model is good — that is chapter 58's job. You need both.
:::

:::check
- `reports/parity.json` says PASS.
- I tested through the gateway as well as against vLLM directly.
:::
