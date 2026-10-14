# 6. Step 3 — No data? Build it from documents and from reality

:::why The usual situation
The company gives you documents, not a training set. Sometimes not even that. Your job is to produce data that behaves like the real thing, not data that looks plausible.
:::

## 6.1 Which source you have decides the route

|What you have|Route|Command|
|---|---|---|
|documents (10-20k files)|ingest → chunk → grounded Q&A generation|`peftkit.data.build` (chapter 12)|
|a small real sample (hundreds of rows)|profile it, then synthesise to match it|`synthkit` (chapter 7)|
|both|use the real sample to **calibrate** the generated data|chapters 7-8|
|neither|expert interviews first: 100 expert-written rows beat 100k generated ones as a seed AND as the gold set|the data chapters|

## 6.2 Documents are not training data

A chunk of a manual is not an example of the behaviour you want. Turn documents into **tasks**:

```text
document chunk ─┬─> grounded Q&A pairs (answer strictly from the chunk)
                ├─> instruction variants (5 phrasings per intent, varied word order)
                ├─> refusals ("that is not in the documents")
                └─> hard cases (ambiguous, contradictory, multi-step)
```

## 6.3 The decision everyone gets wrong

**20,000 documents is usually a retrieval problem, not a fine-tuning problem.** Facts change; behaviour does not. Fine-tune the *behaviour* of using retrieved context (format, tone, refusals, citations); retrieve the *facts*. The chapter on fine-tuning vs RAG has the measurement behind that. When the company wants "our AI brain to switch domains", what they usually need is: retrieval over their corpus **plus** a small behaviour fine-tune.

:::check
- I can say which part of the request is behaviour (fine-tune) and which is facts (retrieve).
- I have a real seed sample, or a plan to get one this week.
:::
