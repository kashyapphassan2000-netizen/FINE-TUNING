# 8. LoRA+: separate learning rates for A and B

:::why In one line
Use a larger learning rate for B than for A (ratio λ = η_{B}/η_{A}, e.g. 16). Same parameters and memory as LoRA, often faster learning. Hayou, Ghosh and Yu, 2024 (arXiv 2402.12354).
:::

## 8.1 Why

In ΔW = B A, B starts at zero and A is random. The paper's scaling analysis for wide models shows that one shared learning rate is inefficient: B needs to move much further than A. Giving B a larger step fixes it. The paper reports faster fine-tuning and small quality gains at the same cost; the exact numbers depend on model and task.

$$ A ← A − η · g_{A} ,     B ← B − λ η · g_{B} ,     λ ≫ 1

## 8.2 Worked example

@file labs/lab6_loraplus.py
@lab lab6_loraplus

At equal steps, ratio 4-16 beat ratio 1 by orders of magnitude on this toy, and ratio 64 was already worse than 4 and 16. The best ratio is task dependent; 16 is a sensible start.

## 8.3 In the template

@file configs/methods/loraplus.yaml

`train.py` builds the optimizer with PEFT's `create_loraplus_optimizer(model, optimizer_cls, lr=..., loraplus_lr_ratio=...)` (signature read from peft 0.21.0) and passes it to the trainer. The trainer still builds the learning-rate schedule from the YAML. `train.lr` is the learning rate of A.

The adapter file is a plain LoRA adapter, so vLLM serves it directly (tested with `serve_check`).

:::check
- I know `train.lr` is η_{A} and B gets λ times more.
- I ran lab 6.
:::
