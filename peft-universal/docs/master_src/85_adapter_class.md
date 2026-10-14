# 85. The UniversalModelAdapter

@file src/uadapt/adapter.py

## 85.1 Method → step → what it automates → what still needs you

|Method|Step|Automated|Needs your judgement|
|---|---|---|---|
|`inspect()`|0|parameter counts by type, signature, inventory|reading `modeling_x.py` when it matters|
|`classify()`|1|family labels with evidence|resolving conflicting evidence|
|`analyze_forward()`|2|traced table: order, shapes, std|interpreting unusual flows|
|`identify_inputs/outputs/loss()`|2-3|signature, output shapes, loss presence and names|choosing the objective for a NEW task|
|`discover_trainable_modules()`|5|roles by computation, targets, exclusions|overriding for domain reasons (e.g. adapt the encoder too)|
|`recommend_peft_strategy()`|4|ranked options from constraints|trading quality vs cost vs latency|
|`build_training_pipeline()`|6|LoRA injection, freezing, head unfreezing|rank/alpha choice (chapter 38)|
|`validate_training(loss_fn, batch)`|8|gradient report: no-grad, zero, NaN, frozen-with-grad|—|
|`train(loader, loss_fn, ...)`|9|AMP, accumulation, clipping, schedule, NaN guard, checkpoints|data, stopping rule, hyperparameters|
|`evaluate(eval_fn)`|11|runs your metric in eval/no-grad|**the metric and the eval set**|
|`export(path, merge=True)`|12|adapter file + merged state dict|verifying merged output (do it)|
|`serve()`|13|suggests an engine per family|the final serving architecture|

## 85.2 Use it

```py
from uadapt.adapter import UniversalModelAdapter
ad = UniversalModelAdapter(model, batch)            # batch WITH labels for encoder-decoders
print(ad.classify()); print(ad.analyze_forward(20))
targets = ad.discover_trainable_modules()
print(ad.recommend_peft_strategy(vram_gb=48, task="classification", new_output_space=True))
ad.build_training_pipeline(r=8, train_also=targets["train_fully_or_freeze"])
assert ad.validate_training(loss_fn, batch)["ok"]
ad.train(loader, loss_fn, lr=2e-4, max_steps=1000, amp_dtype="bf16")
print(ad.evaluate(my_eval_fn)); print(ad.export("export/", merge=True)); print(ad.serve())
```
