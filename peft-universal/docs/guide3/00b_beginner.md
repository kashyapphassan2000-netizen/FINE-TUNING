# 0B. Plain-English primer: read this first if you are new

No maths in this chapter. If you have never thought about how a model writes text, this is everything you need before chapter 1.

## 0B.1 How a model writes a sentence

A language model writes **one token at a time** (a token is a word or a piece of a word). To write the next one, it re-reads everything written so far and asks: what comes next?

```text
prompt: "The pump pressure must"
step 1  -> "stay"          (reads: The pump pressure must)
step 2  -> "below"         (reads: The pump pressure must stay)
step 3  -> "six"           (reads: The pump pressure must stay below)
```

Notice that each step re-reads a longer text. That is the whole problem this book solves.

## 0B.2 The notes analogy (this is the KV cache)

Imagine summarising a two-hour meeting. Every time you write a word, you re-watch the entire recording from the start. Writing 500 words = 500 re-watches. Obviously you would take **notes** once and glance at them instead.

- The recording = all previous tokens.
- The notes = the **KV cache**.
- "K" and "V" are just the two kinds of note the model keeps for each token: **K** ("what I am about", used to decide relevance) and **V** ("what I contribute", the content passed on).

The model computes K and V once per token, stores them, and reuses them for every later token. Same answers, far less work.

## 0B.3 The catch: notes take up space

The notes live in GPU memory and they grow:

- longer conversation → more notes;
- more users at once → more notes;
- more layers in the model → more notes (every layer keeps its own).

For Llama-3.1-8B the notes cost about **128 KB per token**. A 128,000-token conversation therefore needs about **16 GB** of GPU memory for notes alone — often more than the model itself. That is why the rest of this book exists: every technique is a way to make the notes smaller, cheaper, or better organised.

## 0B.4 The thirteen topics, in one sentence each

|Topic|Plain meaning|Chapter|
|---|---|---|
|Naive decoding problem|re-reading everything each step is quadratic work|1|
|KV cache|keep notes so each step is cheap|2|
|KV cache memory math|how many bytes those notes cost, exactly|3|
|FlashAttention|compute attention in small tiles so a huge intermediate table never exists|4|
|PyTorch SDPA|one function call that picks the fastest safe attention kernel for you|5|
|Attention variants|four ways to decide how many notes to keep per token|6|
|MHA|every head keeps its own notes: best quality, biggest memory|7|
|MQA|all heads share one set of notes: smallest memory, some quality loss|8|
|GQA|small groups of heads share notes: the modern default|9|
|MLA|compress the notes and rebuild them on demand: smallest at large scale|10|
|PagedAttention / vLLM|store notes in small fixed blocks so nothing is wasted or duplicated|11|
|Positional encodings|how the model knows the order of words|12|
|RoPE|the rotation trick that encodes position in almost every modern model|13|

## 0B.5 Words you will meet

|Word|Meaning|
|---|---|
|token|a word or part of a word; models read and write these|
|context length|how many tokens the model can look at at once|
|head|attention runs in parallel "heads", each looking for different things|
|layer|the block above is repeated many times; each layer keeps its own notes|
|prefill|processing the prompt (all at once, fast)|
|decode|generating tokens one by one (this is where the cache pays off)|
|VRAM|GPU memory: weights + notes + working space all live here|
|throughput / latency|tokens per second in total / how long one user waits|
|quantization|storing numbers in fewer bits (e.g. fp8) to save memory|

## 0B.6 How to read the rest

Every chapter is: a plain idea → the problem → the maths with a real number → runnable code → what goes wrong → what production does. If a chapter's maths is heavy on a first pass, read the intuition box and the lab output, then move on and come back.

Run the labs as you go. They are small programs that print real numbers in seconds on any laptop, no GPU needed:

```any
python labs/lab30_naive_decoding.py       # ... through lab37_positional_schemes.py
```

:::check You are ready for chapter 1 when
- You can explain the meeting-notes analogy to a colleague.
- You know that the notes (KV cache) grow with tokens, users and layers.
- You have run one lab and seen output.
:::
