# 114. The engine architecture

The review's point 18: evolve from a collection of recipes into an engine. This is that engine, with what is implemented marked.

```text
                     ┌───────────────────────────┐
                     │  MODEL (Hub id / dir / nn.Module) │
                     └──────────────┬────────────┘
                                    ▼
                   ARCHITECTURE ANALYZER   uadapt.inspect_model  (ch 69-70)   [implemented]
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
          MODALITY              ARCHITECTURE            MoE / quantized / hybrid flags
        (text/image/…)       (dec/enc/enc-dec/…)        (ch 81, 22)          [implemented]
              └─────────────────────┬─────────────────────┘
                                    ▼
                    TASK RESOLUTION   uadapt.tasks.resolve / plan   (ch 107)  [implemented]
                                    │
        ┌───────────────┬───────────┼───────────────┬───────────────┐
        ▼               ▼           ▼               ▼               ▼
   DATA ADAPTER     PROCESSOR    COLLATOR      LOSS ENGINE      METRICS ENGINE
   (ch 87)          (per model)  (ch 87)       (tasks.loss_fn)  (uadapt.metrics)
   [rules + code]   [model's own][pattern]     [10 shipped]     [implemented]
        └───────────────┴───────────┼───────────────┴───────────────┘
                                    ▼
                      PEFT ENGINE   targets.discover + lora.inject   (ch 74-75)   [implemented]
                                    ▼
                      TRAINING ENGINE   uadapt.loop / TRL / family Trainer (ch 84) [implemented]
                                    ▼
                      VALIDATION   gradient report + one-batch overfit (ch 71)     [implemented]
                                    ▼
                      EVALUATION ENGINE   baseline → tuned → compare (ch 45-48, 89)[implemented for text;
                                                                                    metrics for all tasks]
                                    ▼
                      CHECKPOINT / MERGE   adapter_state_dict + merge_all + proof   [implemented]
                                    ▼
                      SERVING   engine chosen by architecture (ch 90)               [vLLM + serve_local]
                                    ▼
                      SERVED PARITY → LOAD TEST → RELEASE GATE (ch 54-56)           [implemented]
                                    ▼
                      GATEWAY: keys, limits, metering → AUTOSCALING (ch 96-97)      [implemented]
                                    ▼
                      MONITORING → regenerate data → retrain (ch 19, 113)
```

## 114.1 What is genuinely automatic, and what never will be

|Automatic|Yours forever|
|---|---|
|inspection, classification, tracing|what "good" means for your task|
|target discovery, adapter injection, freezing|the data itself, and its licence|
|gradient checks, NaN guards, one-batch smoke test|the eval set, written by humans|
|training loop mechanics, checkpoints, merging with proof|the serving SLO and the budget|
|metric computation per task|whether the result is worth shipping|

That division is the honest answer to "can this be one button?" — no, and the parts that cannot be automated are exactly the parts that decide whether the model is any good.
