# 66. Plain-English primer for Parts J, K and L — start here if the words are new

No maths in this chapter. Chapters 5-65 taught you one road: take a chat model, fine-tune it, evaluate it, serve it. Parts J, K and L answer three new questions:

- **J: what if the model is not a chat model?** (an image model, a speech model, a model you wrote yourself)
- **K: how many GPUs, from whom, at what cost, and how do strangers use it through an API?**
- **L: can you do all of it alone, on a model nobody has written a tutorial for?**

## 66.1 The idea of Part J in one picture

```text
Every model, no matter what it is, is the same five things:
   INPUT  ->  LAYERS  ->  OUTPUT  ->  a number saying how wrong it was (LOSS)  ->  small corrections
   (text,      (the         (words,       (the score to minimise)                  (training)
    image,      "brain")     labels,
    sound)                   pixels)
```

Fine-tuning any model means answering the same questions, in the same order:

|Question|Plain meaning|Chapter|
|---|---|---|
|What is this thing?|read its files: what it eats, what it says|68|
|Which family?|chat model? image model? speech? something else?|70|
|What happens inside?|watch one example go through it|71|
|What is it scored on?|which "how wrong was I" number it uses|72|
|What should I change?|everything, or a tiny part?|73|
|Which layers exactly?|found by what they compute, not by their names|74|
|The library does not support it — now what?|write the missing piece yourself|75|

## 66.2 Words you will meet in Part J

|Word|Plain meaning|
|---|---|
|architecture|the shape of the model: which layers, in which order|
|decoder-only|a model that writes text one word at a time (ChatGPT-style). Parts A-I are about these|
|encoder-only|a model that reads text and outputs a label or a vector (e.g. "is this email spam?")|
|encoder-decoder|reads one thing, writes another (translation, summarisation, speech-to-text)|
|vision model|takes images. **CNN** and **ViT** are two common designs|
|CTC|a way to train speech models when you know WHAT was said but not exactly WHEN|
|multimodal|takes two kinds of input at once, e.g. an image plus a question|
|MoE (mixture of experts)|a big model where only a few "experts" work on each word: huge memory, modest compute|
|SSM / Mamba|a newer design that keeps a small running memory instead of looking back at everything|
|RNN / LSTM|an older design that reads step by step, keeping a memory|
|diffusion|image generators: they start from noise and clean it up step by step|
|forward pass|running one example through the model|
|hook|a small spy you attach to a layer to see what goes in and out|
|gradient|"if I nudge this number, how much does the error change?" No gradient = that part is not learning|
|adapter / LoRA|a small add-on you train instead of the whole model|
|merge|folding the add-on back into the model so it becomes one normal model again|
|head|the last layer, which produces the final answer (labels, words, boxes)|

## 66.3 The idea of Part K in one picture

```text
users  ->  requests per second  ->  how many run at the same time  ->  how many servers you need  ->  cost
                                  (Little's law: concurrency = rate x how long one answer takes)
```

|Word|Plain meaning|Chapter|
|---|---|---|
|VRAM|memory on the GPU. The model's weights live there, plus the "notes" for each user|92|
|replica|one running copy of your model serving requests|93|
|tensor parallel|splitting ONE copy across several GPUs because it does not fit on one|93|
|latency / p95|how long one answer takes / the slow 5% of answers|55|
|throughput|how many requests per second the server handles|55|
|cold start|the delay before a brand-new server can answer: it must download and load the model|94|
|autoscaling|adding servers when users arrive and removing them when they leave|97|
|API key|a secret string a user sends so your service knows who they are and can limit them|96|
|gateway|the doorman in front of the model: checks keys, counts usage, enforces limits|96|
|metering|counting each user's tokens — what a bill would be based on|96|

## 66.4 What "anyone can use my model" actually means

```text
your customer's code                your service
  OpenAI SDK, base_url=your URL  ->  GATEWAY (checks their key, their limit, their permissions)
                                        -> vLLM (the model on a GPU, answering many users at once)
                                        -> counts their tokens
                                     autoscaler watches the queue and adds/removes GPUs
```

Chapter 96 has this working: the rehearsal issues a key to a new customer and the customer gets an answer.

## 66.5 The honest part

- Free hosting is for demos and for you alone. Serving strangers needs a paid GPU (chapter 98).
- "How many GPUs" cannot be answered from a formula alone: you measure one server with a load test first (chapter 55), then multiply (chapter 93).
- A model that is not a chat model usually cannot be served by vLLM. You choose the engine from the architecture (chapter 90).

## 66.6 How to read the rest

Read chapter 67 (what generalises and what does not), then 68-75 in order — they are the method. Chapters 76-83 are one architecture family each: read the one you have. Then Part K when you want users, and chapter 100 when you want to test yourself.

:::check You are ready for chapter 67 when
- You can say what a forward pass, a gradient and an adapter are, in your own words.
- You can explain why more users need more GPUs, and what a cold start is.
:::
