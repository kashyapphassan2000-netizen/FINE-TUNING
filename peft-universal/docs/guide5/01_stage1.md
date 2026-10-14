# 1. SFT as stage 1 of every post-training pipeline

:::why In one line
Pretraining gives a model language and knowledge. SFT is the first stage that makes it **answer** — in your format, for your task, with your behaviour. Everything after it assumes it happened.
:::

## 1.1 The pipeline

```text
pretraining            trillions of raw tokens      the model learns language and facts
   │
[continued pretraining]  your raw domain text       new vocabulary, domain patterns (optional)
   │
SUPERVISED FINE-TUNING   prompt -> ideal answer     follows instructions, in your format   <-- this book
   │
preference optimization  chosen vs rejected         picks the better of two good answers
   │
evaluation + iteration   real held-out data         the only evidence that any of it worked
```

- Only the **assistant** tokens are trained (Volume 2, chapter 4). The prompt is context, not a target.
- SFT is where the chat template, the system prompt behaviour and the output format are learned.
- DPO/ORPO on a model that has not been SFT'd mostly amplifies noise: there is no baseline behaviour to prefer between.

## 1.2 What SFT changes, and what it cannot

@file labs/lab50_sft_stage1.py
@lab lab50_sft_stage1

Two results in that output, both worth internalising:

1. **It learns the behaviour that is in the data.** Accuracy on the trained prompts goes from chance to 1.0.
2. **It generalised to nothing.** Accuracy on unseen prompts is 0.0 — with six training examples the model memorised. That is not a bug in SFT; it is what six examples buy. Chapter 5 measures how much data starts to buy generalisation.
3. **It cannot invent facts.** Asked about a case absent from the data, the tuned model answers confidently in the learned **format** with a guess. Format is learned; facts come from pretraining, retrieval, or data that actually contains them.

## 1.3 When SFT is the wrong tool

|Situation|Use instead|
|---|---|
|the facts change weekly|retrieval (Volume 2, chapter 16)|
|you need a brand-new language or field|continued pretraining first|
|both answers are fine, one is just better|preference optimization after SFT|
|the base model cannot do the task at all|a stronger base model|
|you have 50 examples and no eval set|write the eval set first, then revisit|

## 1.4 The minimum viable SFT project

1. 50-item human-written eval set, built **first**, never trained on.
2. 300+ training examples covering every intent and phrasing you expect.
3. The model's own chat template, with assistant-only loss.
4. LoRA r=16, lr 2e-4, 1-3 epochs, eval every 20-50 steps.
5. Compare against the base model on the eval set. If it did not move, the data is the problem.

:::check
- I can place SFT in the pipeline and say what each neighbouring stage needs from it.
- I know SFT teaches format and behaviour, not facts.
- My eval set exists before my training set.
:::
