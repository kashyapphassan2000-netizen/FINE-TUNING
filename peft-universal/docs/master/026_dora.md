# 26. DoRA: magnitude + direction decomposition

:::why In one line
Split each weight matrix into a length (magnitude per column) and a direction. Train the magnitude directly, and train the direction with LoRA. Liu et al., 2024 (arXiv 2402.09353).
:::

## 26.1 Intuition

A weight column is an arrow. Full fine-tuning changes arrows' lengths and directions with little coupling. The DoRA paper's analysis found LoRA's updates couple the two. DoRA gives length its own parameter, so LoRA's rank is spent only on direction.

## 26.2 The math

$$ W = m · V / ‖V‖_{c}      (m = ‖W‖_{c}: one number per column)

$$ W' = m · (W_{0} + B A) / ‖W_{0} + B A‖_{c}

- m starts as the column norms of W_{0}, so W' = W_{0} at init.
- Trainable: A, B (as LoRA) + m (d_{out} or d_{in} numbers depending on the convention; PEFT's count for our tiny model was exactly LoRA + sum of output sizes).
- The paper treats ‖·‖_{c} as a constant in backward to save memory; PEFT does the same (detached norm).

## 26.3 Worked example

@file labs/lab4_dora.py
@lab lab4_dora

The target changes only the lengths of 8 columns. A rank-4 LoRA cannot express 8 independent column rescales; DoRA's magnitude vector can, so its loss is ~12x lower.

## 26.4 In the template

@file configs/methods/dora.yaml

`peft_factory.py` sets `LoraConfig(..., use_dora=True)`.

## 26.5 Costs and limits

- Slower and more memory than LoRA (extra norm computation per step). The estimator adds a rough 30% to activation memory; measure yours.
- **vLLM 0.29.0 rejects DoRA adapters** (its `peft_helper.py`: "vLLM does not yet support DoRA"). Merge before serving. After merging, a DoRA model is a normal model with zero overhead.
- Merge was exact in our tests (max logit difference 3.7e-7).

:::check
- I can write W' = m · V/‖V‖_{c}.
- I know DoRA must be merged for vLLM.
:::
