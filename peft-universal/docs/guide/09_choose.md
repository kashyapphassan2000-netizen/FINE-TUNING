# 9. Choosing a method + the universal hyperparameter sheet

## 9.1 Decision table

|Situation|Method|Why|
|---|---|---|
|Model fits in bf16 with room to spare|LoRA|fastest, simplest, servable as adapter|
|Model does not fit in bf16 on your GPU|QLoRA|~4x smaller base|
|You want max quality at LoRA size, can merge|DoRA (or QDoRA)|decouples magnitude/direction|
|Long training, unsure where capacity is needed|AdaLoRA|learns per-layer rank|
|Same cost as LoRA, want faster convergence|LoRA+|B learns faster|
|Many tenants on one base model|LoRA / LoRA+ / QLoRA adapters|vLLM multi-LoRA|

## 9.2 Starting hyperparameters (then tune on your eval set)

|Setting|Start|Notes|
|---|---|---|
|r / alpha|16 / 32|raise r for knowledge-heavy tasks|
|dropout|0.05|0 for very large datasets|
|lr|2e-4 LoRA/QLoRA, 1e-4 DoRA, 1e-4 (A) LoRA+|halve if loss spikes|
|scheduler / warmup|cosine / 3%|`warmup_steps: 0.03` (float = ratio in transformers 5)|
|effective batch|16-64 sequences|per_device × grad_accum × GPUs|
|epochs|1-3|watch eval loss; more epochs overfit small data|
|max_seq_length|p95 of your token lengths|longer = more memory|
|gradient checkpointing|on|~30% slower, much less memory|
|packing|off to start|on for many short examples; not with AdaLoRA auto steps|

## 9.3 Small model → big model: what changes

|Knob|135M laptop test|8B on 1× 48 GB|70B|
|---|---|---|---|
|method|lora|lora or qlora|qlora (1× 80 GB) or lora + FSDP2|
|per_device_batch_size|8|2|1|
|grad_accum|2|8|16|
|max_seq_length|1024|4096|2048-4096|
|launcher|python|python|accelerate launch|
|serving|1 GPU|1 GPU|tensor parallel over 2-4 GPUs|

The code, data format and commands do not change. Only the model YAML and, for multi-GPU, the launcher change.

## 9.4 PEFT interview questions (15)

1. **Why does LoRA start with B = 0?** So ΔW = 0 and the model starts identical to the base; training begins from the pretrained behaviour.
2. **What does α/r do?** Scales the update so changing r does not change its size much.
3. **Why is LoRA free at inference?** B A can be added into W_{0}; the merged model has the same shape.
4. **Trainable params for a 4096×4096 layer at r=16?** 16 × 8192 = 131,072.
5. **What is NF4?** 4-bit data type with levels at normal-distribution quantiles.
6. **What does double quantization save?** ~0.37 bit/param by quantizing the block scales.
7. **What are paged optimizers for?** Surviving memory spikes by paging optimizer state to CPU RAM.
8. **Does QLoRA train the 4-bit weights?** No; they are frozen. Gradients flow through their dequantized values into LoRA.
9. **DoRA's decomposition?** W = m · V/‖V‖_{c}; m trained directly, V via LoRA.
10. **Why must DoRA be merged for vLLM 0.29?** vLLM rejects DoRA adapters.
11. **What does AdaLoRA prune?** Singular values λ in ΔW = PΛQ, by an EMA-smoothed importance score.
12. **Why does AdaLoRA need a callback in HF Trainer?** The Trainer never calls `update_and_allocate`.
13. **What is LoRA+?** A larger learning rate for B than A.
14. **What is rsLoRA?** Scaling by α/√r so high ranks still learn.
15. **When is full fine-tuning still better?** Large distribution shifts with lots of data (e.g., new language, continued pretraining) where low-rank updates underfit.

## 9.5 Exercises (with solutions)

1. Count LoRA params for Llama-3.1-8B attention-only, r=8. **Solution:** per layer 8·[(4096+4096)+(4096+1024)·2+(4096+4096)] = 8·26,624 = 212,992; ×32 = 6,815,744.
2. Change lab 2 to rsLoRA and r=64; compare final MSE. **Solution:** pass `rslora=True`; the scale becomes 32/8 = 2.0 instead of 16/64 = 0.25, so expect faster progress at the same lr. Run it and check.
3. In lab 3 use block 256; does error rise? **Solution:** expect higher error: one scale covers more values, so a single outlier stretches the grid for more weights. Run it and compare.
4. Make lab 4's target a pure rotation; does DoRA still win? **Solution:** no clear win; the magnitude vector does not help a direction-only change.
5. In lab 5 set target budget to 1 per layer (4 total). **Solution:** the 6-rank layer keeps most, others drop to 0 or 1; loss rises.
6. Find the ratio where lab 6 diverges. **Solution:** raise it until the final loss grows; in the recorded run 64 was already worse than 4 and 16.
7. Run `peftkit.hardware` for your model at seq 8192. **Solution:** activation and logits columns grow linearly with sequence length.
8. Train the tiny model with `method.r=2` vs `32` (tests' overrides). **Solution:** expect r=2 to learn more slowly. Compare `train_loss` in each `run_meta.json`; do not assume, measure.
9. Why is `modules_to_save=["lm_head"]` needed after adding tokens? **Solution:** new token rows must be trained; LoRA alone does not learn new embedding rows.
10. Explain why QLoRA can be slower than LoRA. **Solution:** every forward and backward dequantizes blocks.

## 9.6 One-page cheat sheet

```text PEFT cheat sheet
LoRA     h = W0 x + (a/r) B A x      B=0 init   params/layer = r(d_in+d_out)   merge: W0 + (a/r)BA
rsLoRA   scale a/sqrt(r)             use for r >= 64
QLoRA    NF4 base + LoRA; double quant (0.5 -> 0.127 bit/param); paged_adamw_8bit; merge into bf16 base
DoRA     W' = m (W0+BA)/||W0+BA||_c  +out-dim params; slower; MERGE before vLLM 0.29
AdaLoRA  dW = P diag(E) Q; importance = EMA|w*g| x EMA uncertainty; cubic budget; NEEDS callback; MERGE before vLLM
LoRA+    lr_B = ratio * lr_A (start 16); plain LoRA file -> servable as adapter
Defaults r16 a32 drop.05 lr2e-4 cosine warmup 3% all-linear grad-ckpt on
Order    inspect_model -> hardware -> data -> 20-step dry run -> full run -> merge -> serve_check -> publish -> vllm
```
