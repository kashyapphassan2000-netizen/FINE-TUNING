# 120. The four axes: modality × architecture × task × method

:::why The principle the review asked for, stated as a rule
A PEFT method is not a pipeline. **LoRA tells you nothing** about the data format, the loss, the collator, the metric or the serving stack. Those come from the TASK. Choose all four axes deliberately.
:::

```text
MODALITY      text | image | audio | video | image+text | audio+text | any
ARCHITECTURE  decoder-only | encoder-only | encoder-decoder | vision | audio | multimodal | diffusion | SSM | RNN | custom
TASK          causal LM | masked LM | seq2seq | classification | token classification | regression |
              embedding | reranking | reward modelling | preference | multimodal SFT | ASR (CTC / seq2seq) |
              audio classification | image classification | image generation | TTS | video | custom
METHOD        full | LoRA | QLoRA | DoRA | AdaLoRA | LoRA+ | IA3 | prefix | prompt | adapters | selective | head-only
```

## 120.1 The matrix, with the row that this guide's Parts A-I cover

|Modality|Architecture|Task|Typical method|Chapter|
|---|---|---|---|---|
|text|decoder-only|causal LM (SFT)|LoRA / QLoRA|5-65 (the main path)|
|text|decoder-only|preference (DPO)|LoRA on the SFT model|64|
|text|decoder-only|reward modelling|LoRA + scalar head|122|
|text|encoder-only|sequence classification|LoRA + head|90, 122|
|text|encoder-only|token classification|LoRA + head|122|
|text|encoder-only|embedding / retrieval|LoRA, large batches|122|
|text|encoder-only|reranking (cross-encoder)|LoRA + head|122|
|text|encoder-decoder|seq2seq|LoRA incl. cross-attention|91|
|image|vision|classification|conv/attention LoRA + head|92|
|image|diffusion|image generation|LoRA on cross-attention|123|
|audio|audio|ASR (CTC)|LoRA, frozen conv extractor|93|
|audio|encoder-decoder|ASR (seq2seq)|LoRA both sides|93|
|audio|audio|audio classification|LoRA + head|122|
|text→audio|varies|TTS|model-specific|124|
|image+text|multimodal|multimodal SFT|projector + LM LoRA|94, 125|
|video|vision/custom|video classification|temporal LoRA + head|124|
|any|custom|custom|custom LoRA (`uadapt`)|98|

## 120.2 Why one model appears in several rows

The same BERT weights can be a classifier, a token tagger, a regressor, an embedding model, a reranker, a reward model or a masked-LM continuation. Lab 62 trains six of those from the **same architecture** — proof that the architecture never determined the pipeline; the task did.

@file labs/lab62_task_matrix.py
@lab lab62_task_matrix
