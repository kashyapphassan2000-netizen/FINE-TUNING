# 76. Encoder-only models (BERT, RoBERTa, DeBERTa, custom encoders)

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Chapters 5-65 cover causal decoders only. This chapter is derived from the architecture and exercised in lab 60 (BERT row).
:::

## 76.1 What is different

|Aspect|Decoder LM|Encoder|
|---|---|---|
|attention|causal (sees the past)|bidirectional (sees everything)|
|output|next-token logits|one vector per token, or a pooled vector|
|typical tasks|generation, chat|classification, token tagging, retrieval embeddings, extractive QA|
|head|LM head (pretrained)|task head, usually **new and randomly initialised**|
|chat template / assistant masking|yes|no|
|KV cache / vLLM generation|yes|no (single forward pass per input)|

## 76.2 Pipeline

```py sequence classification with LoRA on the encoder and a fully trained head
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                 target_modules=["query", "key", "value", "dense"])   # names from discovery, not memory
model = get_peft_model(model, cfg)          # SEQ_CLS keeps the classifier head trainable
batch = tok(["the pump leaks"], return_tensors="pt", truncation=True, max_length=256)
batch["labels"] = torch.tensor([2])
loss = model(**batch).loss                   # CrossEntropy over num_labels
```

Without PEFT support, the same with `uadapt`: `inject(model, discovered, train_also=["classifier"])` — the lab-60 BERT row went from loss 1.094 to 0.007 on one batch, merge exact.

## 76.3 Traps

- The head is random: evaluate after training only; a "baseline" of the untuned model is chance.
- `max_length` truncation silently drops the end of long documents; chunk or use a long-context encoder.
- Class imbalance: report per-class recall, not accuracy.
- Serving: a plain forward pass behind an API (batch requests); vLLM's strengths (KV cache, generation) do not apply.
