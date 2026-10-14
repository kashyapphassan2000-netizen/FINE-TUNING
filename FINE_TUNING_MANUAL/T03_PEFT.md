# T03 — Parameter-Efficient Fine-Tuning (PEFT)

**WHERE:** read anywhere; labs on your PC (`FINE-TUNING\peft-universal`, `(.venv)` active).
**GOAL:** know what each method changes inside the model, what each knob means, and how to choose.

---

## 1. The intrinsic dimensionality insight

**Claim (Li et al. 2018; Aghajanyan et al. 2020):** the change a pre-trained model needs for a new task lives in a **tiny subspace**. Train only `d` numbers `θ_d` and map them into the full weight space with a fixed random projection `P`:

```text
θ = θ_0 + P · θ_d          θ_0 = pre-trained weights (frozen), P = random D x d matrix (fixed)
d90 = smallest d that reaches 90% of full fine-tuning quality
```

Aghajanyan et al. measured `d90` ≈ **200** parameters for RoBERTa-large on MRPC, out of 355 million. Bigger pre-trained models have **smaller** intrinsic dimension: pre-training compresses the task space. That is the theoretical reason PEFT works and gets **easier** as models grow.

```powershell
python labs/lab1_intrinsic_dimension.py
```

**YOU SHOULD SEE (real output):**

```text
total parameters D = 1474
full training accuracy       = 1.000
d =    2  ( 0.14% of D)  accuracy = 0.491
d =   25  ( 1.70% of D)  accuracy = 0.654
d =  100  ( 6.78% of D)  accuracy = 0.821
d =  200  (13.57% of D)  accuracy = 0.978  <- reaches 90% of full
```

(Toy MLP from scratch, so d90 is a large fraction; on a real pre-trained 8B model it is a vanishing fraction.)

## 2. LoRA: maths, rank, alpha, target modules

**LoRA** (Hu et al. 2021): freeze the weight matrix `W₀` (d_out × d_in) and learn a **low-rank** update.

```text
h = W₀·x + (α / r) · B·A·x          A: r x d_in  (random init: Kaiming-uniform in PEFT)
                                    B: d_out x r (init ZERO -> the model is unchanged at step 0)
trainable per matrix = r · (d_in + d_out)        instead of d_in · d_out
merged weight W' = W₀ + (α / r) · B·A            (exact: no inference cost after merging)
```

Example, one 4096 × 4096 attention projection, r = 16: full = 16.8M parameters; LoRA = 16 × 8192 = **131k (0.78%)**.

| Knob | What it does | Start with | Tune when |
|---|---|---|---|
| `r` (rank) | capacity of the update | 16 | 8 for small/simple tasks; 32-64 for big domain shifts; 64-256 for CPT |
| `alpha` (α) | scale of the update: effective scale = α / r | α = 2r (32 for r = 16) | keep α/r fixed when you change r, or the lr effectively changes |
| `use_rslora` | scale = α / √r instead of α / r (rank-stabilised) | off | on if you sweep large ranks (r ≥ 64): stops the update shrinking as r grows |
| `dropout` | dropout on the LoRA path | 0.05 | 0.1 on tiny datasets; 0 for very large ones |
| `target_modules` | which Linear layers get adapters | `all-linear` (q,k,v,o,gate,up,down) | attention-only on MoE models (otherwise every expert gets adapters) |
| lr | learning rate of A and B | **2e-4** | 1e-4 - 5e-4. **2e-5 is a FULL fine-tuning lr and is too low for LoRA** |

```powershell
python labs/lab2_lora_math.py
```

**YOU SHOULD SEE (real output):**

```text
1) output identical at init: True                     <- B = 0
2) trainable params: full=262,144  lora=8,192  (3.12%)
4) singular values of learned delta (top 10): [27.357, 24.93, 24.67, 22.131, 0.207, ...]
   rank(delta) <= r by construction: 8 <= 8
5) merge is exact: True
6) r= 16: classic scale alpha/r=1.000   rsLoRA scale alpha/sqrt(r)=4.000
```

**Target-module names differ by architecture**: Llama/Qwen/Mistral use `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`; GPT-2 uses `c_attn c_proj c_fc`; Falcon uses `query_key_value dense`; DeepSeek MLA uses `q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj o_proj`. Never guess: `peftkit.plan` prints `LoRA target names` for any model, and `all-linear` finds them automatically.

## 3. QLoRA: 4-bit NF4, double quantization, paged optimizers

**QLoRA** (Dettmers et al. 2023) = LoRA adapters (bf16) on top of a **frozen base stored in 4 bits**. Three ideas:

1. **NF4 (4-bit NormalFloat):** pre-trained weights are roughly normally distributed. NF4's 16 levels are the **quantiles of a normal distribution** (more levels near 0, where most weights are), so every level is used equally often: information-theoretically optimal for normal data. Each block of 64 weights stores one fp32 `absmax` scale.
2. **Double quantization:** the `absmax` constants themselves (32 bits / 64 weights = 0.5 bit/param) are quantized to 8 bits in blocks of 256 → **0.127 bit/param**. Saves 0.373 bit/param (≈3 GB on a 65B model).
3. **Paged optimizers:** optimizer state lives in NVIDIA unified memory; on a memory spike (a long sequence), pages move to CPU RAM instead of crashing with OOM.

Compute still happens in bf16: each 4-bit block is **dequantized on the fly** for the matmul. That is why QLoRA is ~20-40% **slower** than LoRA per step but uses ~1/3 of the weight memory.

```powershell
python labs/lab3_nf4_quantization.py
```

**YOU SHOULD SEE (real output):**

```text
our NF4 codebook : [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, ...]
bitsandbytes NF4 : [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.091, 0.0, 0.0796, ...]
match (atol 1e-4): True
NF4           mean abs error = 1.457e-03
uniform int4  mean abs error = 1.707e-03
quantization constants: 0.500 bit/param without double quant, 0.127 with it
```

**QLoRA traps that cost money:**
- bitsandbytes quantizes **`nn.Linear` only**. Fused MoE expert tensors (3-D parameters), embeddings, norms and conv layers stay 16-bit, and `prepare_model_for_kbit_training` **upcasts** non-quantized tensors to fp32. On MoE models QLoRA can be **bigger** than bf16 LoRA (the planner shows this: DeepSeek-V3 QLoRA = 1.36 TB). `quant.upcast_non_quantized: false` keeps them in bf16.
- **Merge into the bf16 base, never into the 4-bit one** (`peftkit.merge` does this): merging into 4-bit would double-quantize the error.
- QLoRA needs an NVIDIA GPU (bitsandbytes CUDA kernels); it does not run on a CPU laptop.

## 4. DoRA: magnitude + direction decomposition

**DoRA** (Liu et al. 2024) splits each weight into **magnitude** (a vector, one number per output column) and **direction** (unit-norm columns), and applies LoRA only to the direction:

```text
W' = m ⊙ (W₀ + B·A) / ‖W₀ + B·A‖_c        m: trainable magnitude vector (init ‖W₀‖_c)
                                           ‖·‖_c: column-wise L2 norm
```

Full fine-tuning tends to change magnitude and direction independently; plain LoRA couples them. DoRA restores that freedom at the cost of one vector per layer and ~20-30% more compute.

```powershell
python labs/lab4_dora.py
```

**YOU SHOULD SEE (real output):**

```text
1) identical to W0 at init: True
2) trainable: lora=384  +magnitude=64
3) column-rescale task, rank 4:  LoRA loss=0.0092   DoRA loss=0.000734
   (8 rescaled columns need rank 8 for LoRA; DoRA's magnitude vector handles it directly)
```

Use DoRA at low ranks (r = 4-16) when LoRA under-fits; lr usually a bit lower (1e-4). **DoRA adapters must be merged before vLLM serving** (vLLM's multi-LoRA serves plain LoRA only).

## 5. AdaLoRA: adaptive rank allocation

**AdaLoRA** (Zhang et al. 2023) parameterises the update like an SVD and **moves rank budget** to the layers that need it:

```text
ΔW = P · Λ · Q          P: d_out x r, Λ: diagonal r (singular values), Q: r x d_in
importance of each singular value = smoothed (|value x gradient|) with uncertainty (EMA, beta1, beta2)
schedule: warm-up (tinit) -> prune least important every deltaT steps until avg rank = target_r -> freeze (tfinal)
loss += orth_reg_weight · (‖PᵀP − I‖² + ‖QQᵀ − I‖²)    keeps P and Q orthogonal
```

```powershell
python labs/lab5_adalora.py
```

**YOU SHOULD SEE (real output):**

```text
step    0 budget=32 loss=1.1771 ranks=[8, 8, 8, 8]
step  600 budget=15 loss=0.5920 ranks=[6, 2, 5, 2]
step 1499 budget=12 loss=0.2110 ranks=[6, 1, 3, 2]
true ranks : [6, 1, 3, 2]
final ranks: [6, 1, 3, 2] (same total budget, redistributed by importance)
```

**Practical:** needs long runs (hundreds of steps or more: the doctor warns under 200 steps and FAILs an impossible schedule). Hugging Face's Trainer does **not** call AdaLoRA's rank allocator by itself: the toolkit's `AdaLoraCallback` does. Its loss includes the orthogonality penalty, so compare **accuracy/eval scores**, not raw loss. Merge before serving.

## 6. LoRA+: separate learning rates for A and B

**LoRA+** (Hayou et al. 2024): with one learning rate, A and B are updated inefficiently in wide models. Give **B** a larger lr: `lr_B = λ · lr_A`, λ ≈ 16 (2⁴) as a starting point.

```powershell
python labs/lab6_loraplus.py
```

**YOU SHOULD SEE (real output):**

```text
lr_B/lr_A =   1: loss after 300 steps = 1.624e-05
lr_B/lr_A =   4: loss after 300 steps = 3.728e-08
lr_B/lr_A =  16: loss after 300 steps = 1.096e-07
lr_B/lr_A =  64: loss after 300 steps = 3.497e-06
Too large a ratio can destabilise training; the LoRA+ paper suggests ~16 as a starting point.
```

Same adapter format as LoRA (servable as-is on vLLM). In the config, `train.lr` is the lr of A.

## 7. Choosing a method (measured on the same data, same steps)

```powershell
python labs/lab19_method_comparison.py
```

**YOU SHOULD SEE (real output):**

```text
method     trainable  train loss  eval loss  eval acc    sec  adapter MB  vLLM adapter
lora          16,384      4.9065     4.7407     0.978    4.6       0.090  yes   merge diff 5.1e-07
dora          17,408      4.9935     4.8276     0.948    6.2       0.100  no (merge first)
adalora       24,744      5.4390     4.8722     0.933    5.3       0.100  no (merge first)
loraplus      16,384      5.2852     5.0743     0.585    5.0       0.090  yes
qlora     skipped on CPU: 4-bit needs an NVIDIA GPU
```

| Your situation | Method | Why |
|---|---|---|
| default, model fits in bf16 | **LoRA r=16, all-linear, lr 2e-4** | fastest, servable as multi-LoRA, best-understood |
| model does not fit in bf16 on your GPU | **QLoRA** | ~3× less weight memory; merge into bf16 afterwards |
| LoRA under-fits at the rank you can afford | DoRA | better at low rank; merge before serving |
| many layers, unclear where capacity is needed, long run | AdaLoRA | redistributes rank; merge before serving |
| slow convergence on a wide model | LoRA+ | faster convergence at no memory cost |
| small model (≤ 3B), lots of data, budget for 16 B/param | full fine-tuning | highest ceiling; highest forgetting risk |

**The rule:** run the comparison on **your** model and **your** data (P12 does it with the HPO tool). Do not trust a table from a blog, including this one.

---

## CHECKPOINT

- [ ] I can write LoRA's forward pass and count its parameters for a 4096×4096 matrix at r=16.
- [ ] I know why B starts at zero and why α/r matters when I change r.
- [ ] I know the 3 ideas of QLoRA and the 3 QLoRA traps.
- [ ] I can say which methods need merging before vLLM serving (DoRA, AdaLoRA).
- [ ] I ran labs 1-6 and 19.
