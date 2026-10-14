# 54. Load testing and capacity: measure before you scale

:::why In one line
Autoscaling thresholds must come from a measurement of *your* model on *your* GPU with *your* prompts. Ramp concurrency, find the highest level that still meets the latency SLO, and derive the scaling threshold from it.
:::

## 54.1 The tool

@file src/peftkit/loadtest.py

For each concurrency level it records p50/p95/p99 latency, requests/s, output tokens/s, error rate and — when you give it the server's `/metrics` — the queue depth the server reports. Then it recommends:

- `max_concurrency_per_replica`: the highest tested level meeting the SLO;
- `suggested_scale_up_threshold`: the queue depth observed at that level (what KEDA / the scaler should trigger on);
- replicas needed = peak concurrent users ÷ per-replica capacity.

## 54.2 Rehearsal numbers, read honestly

From chapter 2's rehearsal against the local server:

|concurrency|p50 s|p95 s|req/s|
|---|---|---|---|
|1|0.047|0.055|19.9|
|2|0.088|0.106|22.1|
|4|0.159|0.250|21.9|

Latency roughly doubles with each doubling of concurrency while throughput stays flat: the local server generates one request at a time. That is the signature of a server with no batching, and it is exactly what vLLM's continuous batching fixes. **These numbers are for the rehearsal only. Re-run the load test against vLLM on the GPU — those are the numbers that set your autoscaling.**

## 54.3 How to run a meaningful load test

- Use prompts from your real traffic or eval set, not "hi".
- Set `--max-tokens` to your real answer length; decode length dominates latency.
- Test up to the level where the SLO breaks; a test that never fails tells you nothing about the ceiling.
- Run it at the context length and quantization you will ship.
- Re-run after every model, GPU, vLLM version or `max_model_len` change.

:::check
- `reports/loadtest.json` recommendation says `meets_slo: true`.
- I know my per-replica capacity and the queue depth that should trigger scale-up.
:::
