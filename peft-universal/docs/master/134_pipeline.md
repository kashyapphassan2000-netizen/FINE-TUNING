# 134. The modality-independent pipeline

@file src/uadapt/pipeline.py

## 134.1 One object, the whole algorithm

```py
from uadapt.pipeline import UniversalPipeline
p = UniversalPipeline(model, sample_batch, model_id="org/model")   # + task=..., pairs=True, loss_fn=...
p.plan()            # 1-7: modality, contracts, loss, targets, processor, task, metrics; and what is MISSING
p.adapt()           # adapter strategy per layer type, with graceful degradation
p.smoke_test()      # gradients reach the adapters AND one batch overfits
p.fit(loader)       # the modality-independent loop (AMP, accumulation, clipping, NaN guard, checkpoints)
p.evaluate(fn)      # your metric; p.metrics_to_report() lists what the task demands
p.export("out/")    # adapter + merged weights, with the merge difference measured
p.serving_plan()    # the engine for this architecture
p.report()          # everything above, as one dict for your run record
```

## 134.2 The universal collator

@file src/uadapt/collate.py

Per key, with no modality-specific code: variable-length tensors are padded (labels with −100, ids with `pad_id`, floats with 0) and an `attention_mask` is built; equal-shaped tensors are stacked; scalars become tensors; **nested dicts collate recursively**, which is how paired tasks (anchor/positive, chosen/rejected) work without a second code path. `make_loader(records, processor, label_fn)` turns human records straight into batches.

## 134.3 The adapter layer, with graceful degradation

```text
Linear / Conv (groups=1)              -> LoRA wrapper (exact merge)
head / new output space               -> trained fully
router, embeddings, norms, narrow projections -> excluded by rule
packed parameters (LSTM/GRU cells, nn.MultiheadAttention in_proj, fused MoE experts)
                                      -> LoRA cannot reach them: SELECTIVE UNFREEZING, reported as a degradation
nothing adaptable at all              -> the pipeline says so rather than training nothing
```

A test asserts the degradation path: a model whose only parameters live inside an `nn.LSTM` still ends with `trainable > 0` and the message *"selective unfreezing of packed-parameter modules: LoRA could not reach them."*

## 134.4 What the pipeline refuses to invent

`plan()["missing"]` lists exactly what a human must supply for this model: `loss_fn` when no objective is implied, `processor.encode` when no processor exists. Those two, plus your data and your definition of success, are the permanent boundary of automation (chapter 128.1).
