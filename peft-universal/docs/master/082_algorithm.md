# 82. The universal adaptation algorithm

## 82.1 The one distinction that matters

```text
UNIVERSAL METHODOLOGY   ≠   UNIVERSAL CODE

the questions are always the same            the answers need model-specific code:
  what goes in? what comes out?                a processor, a collator, a loss,
  where is the loss? what should train?        an adapter, a trainer, a server
  how do I prove it learned? how do I ship?
```

`uadapt` automates the **questions and the mechanics** (inspection, tracing, target discovery, adapter injection, gradient checks, merging, a training loop). It cannot automate the **judgement**: what the objective should be for your task, what the data must look like, what "good" means, and which serving stack fits.

## 82.2 The algorithm

```text
 0 IDENTIFY      what is this?            files, config, forward signature, module tree        ch 83
 1 CLASSIFY      which family?            evidence -> decision tree                             ch 84
 2 TRACE         what flows through it?   hooks: shapes, dtypes, order, shared inputs           ch 85
 3 OBJECTIVE     what is optimised?       loss in source / outputs.loss / write your own        ch 86
 4 STRATEGY      what should train?       constraints -> full / LoRA / QLoRA / IA3 / head ...    ch 87
 5 TARGETS       which layers exactly?    computation-based discovery                           ch 88
 6 ADAPT         inject + freeze          PEFT if supported, else custom LoRA                   ch 89
 7 DATA          raw -> batch             processor, collator, labels, masks                    ch 101
 8 SMOKE TEST    does it learn at all?    gradient check + overfit ONE batch                    ch 85, 87
 9 TRAIN         the real run             family Trainer or uadapt.loop                         ch 98
10 DEBUG         symptom -> fix           the universal table                                   ch 102
11 EVALUATE      better than baseline?    same unseen suites, before vs after                   ch 103
12 EXPORT        adapter or merged        exact-merge proof                                     ch 89
13 SERVE         which engine?            by architecture, not by habit                         ch 104
14 LOAD TEST     how many users?          SLO -> capacity -> replicas                           ch 68, 92
15 DEPLOY        public API               keys, limits, metering, autoscaling                   ch 110-110
```

## 82.3 The rule that replaces "unsupported"

Never conclude "this model is unsupported, so it cannot be fine-tuned". Ask *why*, and walk the chain:

```text
No PEFT support
  -> Are there Linear / Conv modules?            (inventory, ch 83)          no -> selective unfreezing
  -> Can adapters be injected around them?       (uadapt.lora.inject)
  -> Are the base weights frozen?                (param_counts: trainable)
  -> Do gradients reach the adapters?            (gradient_report)           no -> the layer is unused
  -> Are the wrappers actually used?             (verify_called + grads)     no -> adapt another layer
  -> Can adapter weights be saved alone?         (adapter_state_dict)
  -> Can they be merged exactly?                 (merge_all + diff check)    no -> ship base + adapter
```

Only when a step fails for a reason no engineering can fix is the answer **IMPOSSIBLE WITH CURRENT APPROACH** — and then the reason is written down.

## 82.4 Vocabulary of honesty used in Part J

|Marker|Meaning|
|---|---|
|IMPOSSIBLE WITH CURRENT APPROACH|no amount of code in this approach solves it|
|CUSTOM ENGINEERING REQUIRED|possible, but you write the missing layer yourself|
|FRAMEWORK LIMITATION|the library does not support it; a different library or your own code might|
|NOT ESTABLISHED BY THE PROVIDED PDF|chapters 19-79 do not cover it; derived here from the architecture|
