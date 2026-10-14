# 107. The task registry (the missing abstraction, implemented)

@file src/uadapt/tasks.py

## 107.1 What a task carries

Every registered task holds: **data schema** (what one record must contain), **labels** (how they are built), **loss** (in words, plus a shippable `loss_fn` where one is generic enough), **collator** rules, **PEFT guidance**, **metrics**, **serving note** and honest **notes**. 19 tasks are registered today.

```py
from uadapt.tasks import plan, TASKS, resolve
plan(["encoder-only", "text"])                 # candidates for this architecture, with the first as default
plan(["audio"], "asr_ctc")["card"]             # the full recipe for one task
print(TASKS["embedding"].card())
```

```out REAL OUTPUT
TASK sequence_classification  [text | encoder-only, decoder-only]
  record : {'text': 'str', 'label': 'int'}
  labels : class id
  loss   : cross-entropy (class weights if imbalanced)
  collate: pad to longest; keep the label as a tensor
  adapt  : encoder attention + MLP; classifier head trained fully
  metrics: macro-F1, per-class recall, AUROC (binary), ECE (calibration)
  serve  : single forward pass behind FastAPI/ONNX; batch aggressively
```

## 107.2 The metrics engine

@file src/uadapt/metrics.py

Shipped and tested: accuracy, macro precision/recall/F1 with per-class breakdown, AUROC, **ECE** (calibration), **Recall@K**, **MRR**, **nDCG@K**, **WER**, **CER**, format compliance, distinct-n. `TASK_METRICS` maps every task to the metrics it must report — so "which metric?" is answered by the registry, not by habit.

## 107.3 Executed: ten tasks, one engine

@lab lab62_task_matrix

Read two rows honestly: the seq2seq run reaches exact-match **0.0** and the CTC run **CER 1.0** even though both losses fell hard. That is the entire argument for task metrics — a falling loss is not a result (chapter 44), and on random weights with one batch the metric tells the truth.

## 107.4 Registering your own task

```py
from uadapt.tasks import TrainingTask, register
import torch.nn.functional as F

def my_loss(model, batch):
    return F.smooth_l1_loss(model(batch["x"]).squeeze(-1), batch["y"])

register(TrainingTask(
    name="quality_scoring", modality="text", architectures=["encoder-only"],
    data_schema={"text": "str", "score": "float 0-1"}, labels="float score",
    loss="Huber on the score", loss_fn=my_loss,
    collator="pad to longest; float targets",
    peft="encoder LoRA + regression head trained fully",
    metrics=["MAE", "Spearman correlation with human scores"],
    serving="forward pass; return the score and a confidence band"))
```

The engine then treats it like any built-in task: `plan(["encoder-only"], "quality_scoring")`.
