# 60. Comparing the five methods on YOUR data

:::why
LoRA / QLoRA / DoRA / AdaLoRA / LoRA+ are not ranked. Which wins depends on your data, model, budget and how long you train. Run the table.
:::

## 60.1 The executed comparison

@file labs/lab19_method_comparison.py
@lab lab19_method_comparison

What this specific run shows (tiny model, 4 examples, 30 epochs, r=8, lr 5e-3 for every method):

- **LoRA** best accuracy here, smallest adapter, fastest.
- **DoRA** ~35% slower, slightly more parameters (the magnitude vector), and not servable as an adapter by vLLM 0.29 — merge it.
- **AdaLoRA** more trainable parameters (SVD form at init_r=12) and a loss that includes its orthogonality penalty, so compare **accuracy**, not loss. It also needs long runs to pay off (chapter 27).
- **LoRA+** scored worst *in this run*, and that is informative rather than a verdict: `lr 5e-3` was shared by all methods, and LoRA+ multiplies B's learning rate by 16, so its effective rate was far past the stable range. The fix is to tune lr per method — which is exactly the point of chapter 53.
- **QLoRA** cannot run on CPU; on a GPU expect roughly LoRA's quality at ~a quarter of the base memory, and slower steps.

:::warn Do not copy this table into your project
It is a 120k-parameter random model. Run `labs/lab19_method_comparison.py`'s structure on your model, your data and your eval suite, with lr tuned per method. Comparing methods at one shared learning rate is a common way to publish a wrong conclusion.
:::

## 60.2 The decision, in practice

|Constraint|Choose|
|---|---|
|fits in bf16, want simple and servable|LoRA|
|does not fit in bf16|QLoRA (dense models)|
|quality at LoRA size, merging is fine|DoRA|
|long run, unsure where capacity is needed|AdaLoRA|
|same cost as LoRA, want faster convergence|LoRA+ (tune the ratio and lr together)|
|many tenants on one base|LoRA / LoRA+ / QLoRA adapters (hot-swappable)|

:::check
- I ran at least two methods on my own data with lr tuned for each.
- I compared task score, not just loss.
- I recorded VRAM, time and adapter size alongside quality.
:::
