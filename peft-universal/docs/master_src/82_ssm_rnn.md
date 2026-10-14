# 82. State-space, recurrent and hybrid models

:::warn NOT ESTABLISHED BY THE PROVIDED PDF
Derived from the architectures; exercised in lab 60 (Mamba and custom LSTM rows).
:::

## 82.1 State-space (Mamba-style)

A selective state-space block: `in_proj` (expands to 2× width) → depthwise Conv1d → `x_proj` → `dt_proj` → selective scan → gate → `out_proj`. There is no KV cache; generation keeps a fixed-size state per layer, so memory does **not** grow with context the way attention's does.

- Targets: `in_proj` and `out_proj` (large, capacity-bearing; lab 60 found them as "fused_or_expand" and "down"). `x_proj` was wrapped too; the narrow `dt_proj` is skipped by rule (sensitive).
- The depthwise Conv1d is refused by the conv-LoRA (groups ≠ 1) — by design.
- Lab 60: loss 4.873 → 2.336, merge exact. Fast kernels (`mamba_ssm`) are optional; transformers falls back to a slow path on CPU.

## 82.2 Recurrent (LSTM / GRU)

`nn.LSTM` stores its gates as packed parameters (`weight_ih_l0`, `weight_hh_l0`), not Linear modules. Options:

1. adapt the Linear layers around the cell (lab 60 custom LSTM: input projection wrapped, head trained; loss 3.880 → 2.083);
2. unfreeze the cell itself (it is usually small);
3. re-implement the cell with explicit Linear gates to LoRA it — **CUSTOM ENGINEERING REQUIRED**.

## 82.3 Hybrids

Models mixing attention with SSM or recurrent layers: classify every block (chapter 70), adapt attention blocks as usual and SSM blocks as in 81.1, and size memory from the attention layers only (chapter 22.5). Check `layer_types` in the config; if it is absent, detection is weaker (chapter 20's Falcon-H1 note).
