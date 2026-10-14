# 16. Fine-tuning vs RAG

:::why The one-line rule
Fine-tuning changes **behaviour**. Retrieval supplies **facts**. A fine-tuned model is not a database.
:::

## 16.1 Choosing

|Need|Use|
|---|---|
|new facts, changing often, must be cited|RAG|
|consistent style, tone, format, JSON schema|SFT|
|shorter prompts (behaviour baked in, not repeated)|SFT|
|a small model imitating a big one|SFT (distillation)|
|new vocabulary / whole technical field|CPT, then SFT (ch 17)|
|domain behaviour AND current facts|both: SFT for style, RAG for content|
|answer quality preferences between good answers|preference optimization (ch 18)|

## 16.2 The staleness demonstration

@file labs/lab21_rag_vs_finetune.py
@lab lab21_rag_vs_finetune

Read the honest limits printed in that output. What it proves at any scale: a fact trained into the weights is frozen at training time, and the only way to change it by training is **another training run**. What the toy cannot show — a retrieved fact in the prompt overriding the trained one — needs a model with in-context learning ability; reproduce that row with a real instruct checkpoint before believing it.

## 16.3 The cost difference that decides most projects

|Event|RAG|Fine-tuning|
|---|---|---|
|a document changes|re-index that document (seconds)|re-train and re-deploy|
|add a new product line|add documents|new data, new run, re-evaluate|
|answer must cite a source|natural|not possible reliably|
|prompt is long and expensive|context stays long|shorter prompts after training|
|latency|extra retrieval hop|none|

## 16.4 The usual production answer: both

```text
user question -> retrieve documents -> prompt (system + context + question) -> fine-tuned model -> answer
                                                                     ▲
   SFT taught it: use only the context, answer in our format, say "not in the documents" when it is absent
```

Fine-tune the **behaviour of using context** (including refusing when the context does not contain the answer), and let retrieval carry the facts. That combination is both current and consistent, and it is what most enterprise assistants actually are.

:::check
- I can say in one sentence whether my problem is facts or behaviour.
- If facts: I built retrieval before fine-tuning.
- If both: my training data includes context-grounded examples AND "not in the documents" examples.
:::
