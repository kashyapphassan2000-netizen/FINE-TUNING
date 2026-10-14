# 105. The five support levels: how hard is THIS model?

Before promising anything, place the model on this ladder. It decides how much code you write.

|Level|What you have|What works out of the box|What you must build|Effort|
|---|---|---|---|---|
|**1 Native**|a standard architecture with a standard head (`AutoModelFor*`), a tokenizer/processor, PEFT support|`AutoConfig` → `AutoModel*` → PEFT → Trainer; `peftkit` path for decoder-only text|nothing structural: data and evaluation only|hours|
|**2 Native, unusual modules**|standard loader, but non-standard layer names, fused qkv, MoE experts, pre-quantized weights|loading and training|target discovery (ch 74), maybe a custom collator, memory re-planning (ch 22)|a day|
|**3 Custom model, standard interfaces**|`trust_remote_code` model that returns `loss` and takes tensors|the training loop, if the interface is respected|inspection, custom targets, sometimes custom LoRA, your own processor|days|
|**4 Fully custom architecture**|a plain `nn.Module`; no config, no Trainer, no PEFT|nothing|forward contract, loss, dataset, collator, adapters, loop, checkpointing, export, serving (ch 84)|1-2 weeks|
|**5 Non-transformer generative**|diffusion, TTS, video, audio codecs, world models|little|the whole objective (noise schedules, vocoders, temporal batching), plus sampling-based evaluation and a different serving stack|weeks, with domain knowledge|

**How to place a model in 60 seconds:** run `python -m uadapt.report --model <id>`. Level 1-2 if it loads with an `AutoModelFor*` class and returns a loss; level 3 if it needs `trust_remote_code`; level 4 if there is no config; level 5 if the objective is not "predict the target given the input" (a timestep, a vocoder or a diffusion schedule appears).

:::warn What each level costs in honesty
At level 1-2 you can promise a timeline. At level 3-4 promise a **spike** first: inspect, smoke-test, then estimate. At level 5, do not promise anything before you have reproduced the model's own training objective on a tiny sample.
:::
