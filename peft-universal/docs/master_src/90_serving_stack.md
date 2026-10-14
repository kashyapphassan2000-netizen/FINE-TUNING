# 90. Deployment: choose the serving stack by architecture

:::why In one line
vLLM is excellent for generative language models it supports. It is not a universal inference server. Choose the engine from the architecture, then put the same gateway, keys and autoscaling in front of it.
:::

## 90.1 The path, for any family

```text
training checkpoint -> adapter -> [merged model, if mergeable] -> [quantized, if appropriate and re-evaluated]
  -> inference engine -> API (OpenAI-compatible where it fits) -> served-model evaluation (ch 53)
  -> load test (ch 54) -> release gate (ch 55) -> gateway + keys (ch 96) -> monitoring + autoscaling (ch 97)
```

## 90.2 Which engine

|Family|Engine options|Notes|
|---|---|---|
|decoder-only LLM (incl. MoE, many VLMs)|vLLM (chapter 52); SGLang; TensorRT-LLM|continuous batching, paged KV cache, multi-LoRA; check the model's architecture is in the engine's supported list for your version|
|encoder-only (classification, embeddings)|transformers or ONNX Runtime behind FastAPI; embedding-specific servers|one forward per request; batch aggressively; no KV cache|
|encoder-decoder text|transformers `generate` behind FastAPI; engines with explicit support for that architecture|do not assume vLLM support|
|ASR / audio|framework runtime with chunked/streaming inference; engine support varies by model|streaming changes the API design|
|vision classification / detection|ONNX Runtime / TensorRT / torch.compile behind FastAPI|pre-processing on the server must match training exactly|
|diffusion|pipeline runtime (e.g. a diffusers pipeline) behind a job queue|latency = steps × step time; use async jobs, not synchronous requests|
|custom|`serve_local`-style FastAPI + torch, then optimise (compile, ONNX, TensorRT)|**CUSTOM ENGINEERING REQUIRED**|

The rehearsal in chapter 2 uses `peftkit.serve_local` exactly this way for a model vLLM is not needed for.

## 90.3 What must be re-verified for any engine

- Parity: the engine's output matches offline evaluation (chapter 53).
- Capacity: load test on the engine itself (chapter 54).
- Pre/post-processing identical to training (tokenizer, template, image normalisation, sampling rate).
- The adapter/merged weights that are actually loaded (log the revision and hash at start-up).
