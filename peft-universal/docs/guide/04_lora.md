# 4. LoRA: math, rank, alpha, target modules

:::why In one line
Freeze every weight matrix W_{0}. Learn its change as a product of two thin matrices B·A. Only A and B train. After training, add B·A into W_{0} and the model is as fast as before.
:::

## 4.1 Intuition

A 4096 × 4096 matrix has 16.7 million numbers. A rank-16 change can be written with two matrices of 4096 × 16 and 16 × 4096: 131 thousand numbers. You describe the change with 16 "directions" instead of 16.7 million free numbers.

## 4.2 The math (Hu et al., 2021, arXiv 2106.09685)

For a linear layer with input x:

$$ h = W_{0} x + ΔW x = W_{0} x + (α / r) · B A x

- W_{0} ∈ R^{d_out × d_in}: frozen.
- A ∈ R^{r × d_in}: trainable, random init (the paper uses a Gaussian; PEFT uses Kaiming-uniform).
- B ∈ R^{d_out × r}: trainable, **initialised to zero**. So ΔW = 0 at step 0 and the model starts exactly as the base model.
- r: rank. α (lora_alpha): scaling. The update is scaled by α / r.
- Trainable parameters per layer: r · (d_in + d_out).

After training: W = W_{0} + (α / r) · B A. This is a normal matrix again, so inference has no extra cost (**merging**).

:::math Why α / r
If you double r without changing α, each new direction gets half the scale, so the size of the update stays about the same when you change r. That lets you change r without re-tuning the learning rate much. Common choice: α = 2r (scale 2) or α = r (scale 1).
:::

## 4.3 rsLoRA (Kalajdzievski, 2023)

Classic scaling α / r shrinks the update too much at large ranks, so high ranks learn slowly. rank-stabilized LoRA uses α / √r. In this template: `method.use_rslora: true`. vLLM supports rsLoRA adapters (it reads `use_rslora` and uses α / √r).

## 4.4 Worked numeric example

Layer 4096 × 4096, r = 16, α = 32:

- full: 4096 × 4096 = 16,777,216 parameters
- LoRA: 16 × (4096 + 4096) = 131,072 parameters (0.78%)
- scale: 32 / 16 = 2.0 (rsLoRA would give 32 / 4 = 8.0)

For Llama-3.1-8B with all 7 linear projections per layer targeted, r = 16: 41,943,040 trainable parameters (our estimator; its formula matched PEFT's own count exactly in the tests).

## 4.5 From scratch

@file labs/lab2_lora_math.py
@lab lab2_lora_math

Read the output: at init the LoRA layer equals the base layer; the learned change has at most r non-trivial singular values; the merge is exact.

## 4.6 With the standard library (PEFT)

```py The same thing with PEFT
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
```

In this template the same object is built by `src/peftkit/peft_factory.py` from YAML.

## 4.7 The same math in JAX

LoRA is only matrix algebra, so it works in any framework. Here only A and B are passed to `jax.grad`; W_{0} is a constant.

@file labs/lab7_jax_lora.py
@lab lab7_jax_lora

:::warn A trap this lab found
With plain SGD and A initialised near zero, the model did not learn at all (loss stayed ~1.5). Because B starts at zero, the gradient that reaches B is proportional to A. A tiny A starves B. Fix: A at a normal scale (1/√d), B at zero. PEFT's defaults already do this.
:::

## 4.8 Choosing rank r

|r|Typical use|
|---|---|
|4-8|style/format changes, small datasets (< 5k examples)|
|16|default for most instruction tuning|
|32-64|new domain knowledge, larger datasets, code|
|128+|approaches full fine-tuning; consider rsLoRA|

Rank is a capacity knob. If training loss stops improving while eval is fine, raise r. If eval gets worse while train improves (overfitting), lower r, add dropout, or add data.

## 4.9 Target modules

A Llama-style decoder layer has 7 linear layers: attention `q_proj k_proj v_proj o_proj` and MLP `gate_proj up_proj down_proj`. The original paper adapted only q and v. Later practice (QLoRA paper onward) found that adapting **all linear layers** usually works better for the same parameter budget. This template defaults to `target_modules: all-linear` (PEFT expands it to every linear layer except the output head).

Names differ between architectures. Never guess; print them without downloading weights:

```any Inspect any model's linear layers (downloads only config.json)
python -m peftkit.inspect_model Qwen/Qwen2.5-7B-Instruct
```

|Choice|Params|When|
|---|---|---|
|`["q_proj","v_proj"]`|smallest|very small data, style only|
|attention only (q,k,v,o)|small|light adaptation|
|`all-linear`|~3-4x attention-only|default|
|+ `modules_to_save=["lm_head","embed_tokens"]`|large|you added new tokens (vLLM cannot serve this as an adapter; merge)|

## 4.10 Common mistakes

- Learning rate too low. LoRA typically uses 1e-4 to 3e-4. TRL's default is 2e-5 (made for full fine-tuning); this template sets 2e-4.
- Changing r and keeping α fixed without understanding the scale change.
- Forgetting that `B = 0` at init: if loss does not move at all, check that the adapter is actually attached (`print_trainable_parameters`).
- Serving an adapter with a different base model revision than it was trained on.
- Saving with a local base path: `adapter_config.json` stores `base_model_name_or_path`. If you trained from `/ephemeral/models/x`, set the Hub id when merging (`--base`).

## 4.11 In production

Adapters are small, so one base model can serve many customers: vLLM loads several LoRA adapters on one base and routes by model name (multi-LoRA, chapter 28).

:::check
- I can write h = W_{0}x + (α/r)BAx and name which parts train.
- I can compute trainable parameters for a given layer and rank.
- I ran lab 2 and lab 7.
- I ran `peftkit.inspect_model` on a model I plan to use.
:::
