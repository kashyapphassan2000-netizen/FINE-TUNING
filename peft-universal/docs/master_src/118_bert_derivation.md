# 118. Worked derivation from first principles: BERT, and four others

The second review's example: *"You give it BERT — how do I derive the entire training pipeline from first principles?"* Here is exactly that, produced by the tool rather than written by hand.

@file labs/lab63_universal_pipeline.py
@lab lab63_universal_pipeline

## 118.1 Reading the BERT block of that output

|Step|Derived|How|
|---|---|---|
|1 modality / architecture|`encoder-only, text`|token embeddings, no decoder, classification head|
|2 input contract|`input_ids (4,8) int64`, `labels (4,) int64`|observed from the batch, not assumed|
|3 output contract|last dim **means classes**, rank 2, model returns a loss|width equals `num_labels`|
|4 loss|the model's own cross-entropy, confidence **high**, with "wrong_if: you need a different objective than the head implements"|it returned a loss for a batch with labels|
|5 trainable|13 LoRA targets — `query, key, value, dense` — head trained fully|computation-based discovery (chapter 74)|
|6 processor|none bundled here (random-weight model): the contract lists `input_ids, attention_mask, token_type_ids`|`resolve()`|
|7 task|`sequence_classification` → metrics: macro-F1, per-class recall, AUROC, ECE|registry (chapter 107)|
|verify|gradients OK, 0 unexplained zero-gradients, loss 1.104 → 0.266 on one batch|smoke test|
|export|13 layers merged, max difference **1.2e-07**|merge proof|
|serve|transformers/ONNX behind FastAPI; batch requests — **not** vLLM|architecture-based (chapter 90)|

Nothing in that chain used the name "BERT".

## 118.2 The other four in the same run

|Model|Output means|Derived loss|Targets found|Task|Merge diff|
|---|---|---|---|---|---|
|ViT image classification|classes (rank 2)|model's own CE|`query/key/value/dense` + patch conv|image_classification|~1e-06|
|wav2vec2 CTC|vocabulary (rank 3)|model's own CTC|attention + FFN + conv extractor|asr_ctc|~1e-06|
|T5 seq2seq|vocabulary (rank 3)|model's own CE (teacher forcing)|`q,k,v,o,wi,wo` (32 targets)|seq2seq|~1e-06|
|invented seismic model|"unknown — decide yourself"|**CUSTOM ENGINEERING REQUIRED**|3 targets|custom|~2e-07|

The last row is the important one: when the contract implies no standard objective, the engine **says so** and names what you must supply, instead of picking a plausible-looking loss.

## 118.3 Do it yourself on any model

```any
python -m uadapt.report --model <org/model> --head <auto head>
```

```py or the full derivation, with the rationale for every step
from uadapt.pipeline import UniversalPipeline
p = UniversalPipeline(model, batch, model_id="org/model")
import json; print(json.dumps(p.plan(), indent=2, default=str))
```
