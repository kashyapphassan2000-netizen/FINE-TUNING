# 20. Fine-tuning fundamentals: what actually changes

:::why In one line
Training = make a prediction, measure how wrong it is, compute how each weight contributed to being wrong, and nudge every weight a little in the direction that reduces the error. Repeat.
:::

## 20.1 The four kinds of training (they are not the same thing)

|Kind|Data|Loss on|Changes|Use when|
|---|---|---|---|---|
|Pretraining|trillions of raw web tokens|every token|all weights, from random|you are a frontier lab|
|Continued pretraining (CPT)|your raw domain text (millions+ of tokens)|every token|all weights, or a high-rank adapter|new jargon, new language, whole new field (ch 63)|
|SFT (supervised fine-tuning)|question → answer pairs|answer tokens only|all weights (full FT) or a small adapter (PEFT)|behaviour, format, tone, task skill. **Most projects**|
|Preference optimization (DPO etc.)|chosen vs rejected answers|both answers, relatively|small adapter|"both answers look fine, but this one is better" (ch 64)|

Order in a full pipeline: pretraining → CPT → SFT → preference optimization → evaluation at every stage.

## 20.2 Full fine-tuning vs PEFT

|-|Full fine-tuning|PEFT (LoRA family)|
|---|---|---|
|what trains|every weight|a small added matrix per layer (0.1-2% of weights)|
|memory|weights + gradients + optimizer for ALL parameters (≈16 bytes each)|base frozen; 16 bytes only for the adapter|
|8B model, Adam|≈120 GB before activations|≈20 GB (ch 23 of this book)|
|output|a new 16 GB model per task|a 10-200 MB adapter per task|
|many tasks|many full models|one base + many adapters (chapter 71)|
|forgetting|high|lower, but real (ch 59)|
|when better|huge distribution shift, lots of data, you own the GPUs|almost everything else|

## 20.3 One training step, with real numbers

@file labs/lab11_training_mechanics.py
@lab lab11_training_mechanics

Read it line by line:

1. **Forward pass**: the model computes a prediction from the input. Here `w·x + b = 7`.
2. **Loss**: how wrong that is. `(7 − 10)² = 9`. Lower is better; this single number is what everything optimises.
3. **Backpropagation**: the chain rule walks backwards through the computation and produces a **gradient** for every trainable parameter: "if you increase this weight by a tiny amount, the loss changes by this much". Hand-computed `dL/dw = 2(y − target)x = −12`, and autograd agrees.
4. **Optimizer** applies the update: `w ← w − learning_rate × gradient`. With lr 0.1: `3.0 − 0.1(−12) = 4.2`.
5. **Verification**: the new prediction is closer and the loss dropped from 9 to 3.24.

The **learning rate** is the size of that nudge. Too small: nothing moves (see the 2e-5 row in chapter 52). Too large: the loss explodes to NaN (chapter 54, case 5).

**AdamW**, used by almost everyone, keeps two running averages per parameter (mean and squared mean of gradients) and divides by the second, so the step size is roughly the learning rate regardless of how big the raw gradient is. That is why it needs 8 extra bytes per trainable parameter, and why PEFT's memory saving is so large: those bytes are only paid for the adapter.

## 20.4 Where PEFT changes the picture

The last block of the lab shows it: after backward, the frozen base weight has `grad = None` while the adapter matrices have gradients. No gradient buffer, no optimizer state, no master copy for the base model. The forward pass still uses the base weights — they are read, not updated.

:::check
- I can name the six stages: forward, loss, backward, gradient, optimizer step, verify.
- I can explain why AdamW costs ~12 bytes per trained parameter.
- I ran lab 11 and the hand-computed gradient matched autograd.
:::
