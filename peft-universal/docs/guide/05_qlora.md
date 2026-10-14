# 5. QLoRA: 4-bit NF4, double quantization, paged optimizers

:::why In one line
Store the frozen base model in 4 bits instead of 16, keep LoRA adapters in 16/32 bits, and compute in bf16. The paper (Dettmers et al., 2023, arXiv 2305.14314) fine-tuned a 65B model on a single 48 GB GPU this way.
:::

## 5.1 Three ideas

1. **NF4 (4-bit NormalFloat).** Pretrained weights are roughly normally distributed. NF4 places its 16 levels at quantiles of a normal distribution, so each level is used about equally often. The paper calls it information-theoretically optimal for normally distributed weights.
2. **Double quantization.** Weights are quantized in blocks of 64, each block has one scale (absmax) in 32 bits: 32/64 = 0.5 bit per parameter overhead. QLoRA quantizes those scales again to 8 bits in blocks of 256: 8/64 + 32/(64·256) ≈ 0.127 bit. Saving ≈ 0.373 bit per parameter (about 3 GB for 65B).
3. **Paged optimizers.** Optimizer states use NVIDIA unified memory, so a sudden memory spike (a long sequence) pages to CPU RAM instead of crashing with out-of-memory. In code: `optim: paged_adamw_8bit`.

## 5.2 Math of blockwise quantization

For a block of 64 weights w:

$$ c = max|w| ,     q_{i} = argmin_{j} | w_{i} / c − v_{j} | ,     ŵ_{i} = c · v_{q_{i}}

v_{j} are the 16 NF4 values. Only q (4 bits each) and c are stored. During forward and backward, the block is dequantized to the compute dtype (bf16), multiplied, and discarded. Gradients flow through the frozen dequantized weight into the LoRA matrices; the 4-bit weights never change.

## 5.3 Worked example, checked against bitsandbytes

@file labs/lab3_nf4_quantization.py
@lab lab3_nf4_quantization

Our from-scratch codebook matches bitsandbytes 0.50.2's `create_normal_map()` to 4 decimals. On normally distributed weights NF4 has lower error than a uniform 4-bit grid (1.457e-3 vs 1.707e-3 here).

## 5.4 In the template

@file configs/methods/qlora.yaml

`modeling.py` builds `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)`, loads the model with one full copy per process, and calls `prepare_model_for_kbit_training`.

:::warn Memory trap read from the PEFT source
`prepare_model_for_kbit_training` casts **every non-4-bit parameter to float32**: embeddings, LM head, norms. For a 128k-vocabulary 8B model, embeddings + LM head are ~1.05B parameters, so this costs ~4 GB instead of ~2 GB. The estimator in chapter 23 includes it.
:::

:::warn Merge the right way
Never merge into the 4-bit model and save that. `peftkit.merge` reloads the base in bf16 and merges the adapter into full precision weights. That model is what vLLM serves.
:::

## 5.5 Q-variants of the other methods

`method.load_in_4bit: true` works with `dora` (QDoRA), `loraplus` and `adalora` too. With LoRA+ and an 8-bit optimizer name, the template builds bitsandbytes' `PagedAdamW8bit`.

## 5.6 Mistakes

- bf16 on GPUs without bf16 (T4, V100): use `model.dtype: float16` and `method.quant.compute_dtype: float16`.
- QLoRA on CPU: the template refuses with a clear error (tested).
- Expecting QLoRA to be faster: it is usually slower than LoRA because of dequantization. It is a memory trade, not a speed trade.

:::check
- I can explain NF4, double quantization and paged optimizers in one sentence each.
- I ran lab 3 and saw the codebook match.
- I know to merge into the bf16 base.
:::
