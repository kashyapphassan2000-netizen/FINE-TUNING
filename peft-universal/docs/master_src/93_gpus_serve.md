# 93. How many GPUs to SERVE: from users to replicas

## 93.1 Four quantities, in this order

1. **Per replica, does it fit?** weights + KV cache + workspace ≤ VRAM × 0.9 → the tensor-parallel size (1, 2, 4, 8 GPUs per replica).
2. **Per replica, how much traffic within the latency SLO?** From the **load test** (chapter 54). The KV-memory figure is only an upper bound.
3. **How much traffic do the users generate?** Little's law.
4. **Replicas** = peak concurrency × (1 + headroom) ÷ capacity per replica.

## 93.2 Little's law: users → concurrent requests

$$ λ = users × active share × requests per active user per minute / 60      (requests per second)
$$ L = λ × W      (average requests in flight; W = average latency per request, seconds)

Worked: 2,000 users, 10% active at the busy hour, 2 requests/minute each, 4 s per answer:
λ = 2000 × 0.1 × 2 / 60 = 6.67 req/s; L = 6.67 × 4 = **26.7 concurrent requests** on average.

## 93.3 Replicas

$$ replicas = ceil( L × peak factor × (1 + headroom) / capacity per replica )

Worked, with a measured capacity of 16 concurrent requests per replica, peak factor 3, headroom 30%:
ceil(26.7 × 3 × 1.3 / 16) = ceil(6.5) = **7 replicas** (a test asserts this exact result).

## 93.4 The tool, with the maths printed

@file src/peftkit/capacity.py

@lab capacity_example

Read the warning in that output: without a load test the tool falls back to the KV-memory upper bound (535 concurrent 850-token sequences on one H100 for an 8B model) — memory can hold that many sequences, but not within a 4-second latency. **Always pass `--loadtest reports/loadtest.json`.**

## 93.5 Headroom and why

|Factor|Why|
|---|---|
|peak factor (2-4×)|traffic is not flat; the busy hour decides the fleet|
|headroom (20-40%)|a replica can fail or restart; new replicas take minutes to start (chapter 94)|
|min replicas ≥ 1|scale-to-zero means the first user after idle waits for a cold start|
