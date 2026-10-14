# 100. The automatic compatibility report

@file src/uadapt/report.py

```any
python -m uadapt.report --model org/any-model --head causal-lm        # Hub id or local dir (config only)
python -m uadapt.report --model org/any-model --head seq-cls
python -m uadapt.report --custom my_model.py:MyModel --input "x=float:2,3,32,32"
```

@lab uadapt_report_example

## 100.1 Every field, explained

|Field|Meaning|How it is computed|
|---|---|---|
|Architecture|Python class|`type(model).__name__`|
|Classification|families (chapter 84)|evidence from config, signature, module types|
|Parameter Count / Trainable|total and currently trainable|`numel` over parameters|
|Input Types / Forward Signature|what forward() accepts|`inspect.signature`|
|Output Types|keys and shapes for the sample batch|one no-grad forward|
|Loss|does it return a loss; which loss functions its source uses|output + source scan|
|Module Inventory|counts of Linear/Conv/Embedding/Norm/Attention/Recurrent|module walk|
|Potential LoRA Targets|leaf names of discovered targets (chapter 88)|traced computation|
|Do not adapt|embeddings, norms, routers, narrow projections|rules in chapter 88|
|Train fully or freeze|heads|last linear / vocab / label size|
|Conv candidates|dense convs for conv-LoRA|module type + groups|
|Recommended PEFT|ranked strategies (chapter 87)|constraints|
|Memory Estimate|full FT with Adam, frozen bf16, NF4|params × bytes (activations excluded: use `peftkit.plan` for LMs)|
|Risks|MoE, pre-quantized, no targets, no loss|rules|
|Required Custom Code|what you must write|family + findings|
|Recommended serving|engine family|chapter 104|
