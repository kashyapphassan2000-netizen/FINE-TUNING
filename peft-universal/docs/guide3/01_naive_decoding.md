# 1. The naive decoding problem

:::why In one line
To produce token n, attention needs the keys and values of every earlier token. If you recompute them every step, generating n tokens costs O(n²) work. The KV cache is the fix, and it is why everything else in this book exists.
:::

## 1.1 Intuition

You are writing a summary of a long meeting. Every time you write one word, you re-read the entire transcript from the beginning. Writing 500 words means 500 re-reads. Obviously you should take notes once and glance at them instead. Those notes are the KV cache.

## 1.2 The problem, precisely

Generation is autoregressive: one token at a time, each conditioned on everything before it. A naive implementation calls the model on the whole sequence each step:

```text
step 1: forward("The")                    -> "pump"
step 2: forward("The pump")               -> "must"
step 3: forward("The pump must")          -> "stay"
```

Step i processes i tokens. Total work for n steps:

$$ Σ_{i=1..n} i = n(n+1)/2 ≈ n²/2 token-forwards

With a cache, each step processes exactly **one** new token: n token-forwards.

## 1.3 Measured

@file labs/lab30_naive_decoding.py
@lab lab30_naive_decoding

The tiny model in the lab is so small that Python overhead dominates, so the wall-clock gain is modest and honest. The table is the real story: at 128k tokens the naive scheme does **65,536×** the attention work of the cached one.

## 1.4 The two phases this creates

|Phase|What happens|Bound by|Batch shape|
|---|---|---|---|
|**Prefill**|the whole prompt is processed once; K and V for every prompt token are written to the cache|compute (big matmuls)|one long sequence|
|**Decode**|one token at a time; one new K/V appended per step|memory bandwidth (weights + cache are re-read every step)|many sequences, one token each|

Almost every serving decision follows from that split: continuous batching, chunked prefill, prefill/decode disaggregation, and why decode throughput scales with memory bandwidth rather than FLOPs.

## 1.5 Common mistakes

- Benchmarking generation with `use_cache=False` and concluding the model is slow.
- Calling the model on the full sequence each step in custom decoding loops (a very common hand-rolled bug).
- Assuming the cache changes results. It does not: the lab asserts the generated tokens are identical.
- Forgetting that training does **not** use the cache (`use_cache=False` during fine-tuning; it wastes memory and conflicts with gradient checkpointing).

:::check
- I can explain why naive decoding is quadratic.
- I know which phase is compute-bound and which is bandwidth-bound.
- I ran lab 30 and saw identical outputs with and without the cache.
:::
