# 73. Step 4 — Decide what to train

:::why In one line
Pick the smallest trainable set that can express the change your task needs, within your memory and serving constraints. Not "LoRA because everyone uses LoRA".
:::

## 73.1 The options, what each changes, and when

|Strategy|What trains|Choose when|Avoid when|PEFT class (verified in peft 0.21)|
|---|---|---|---|---|
|full fine-tuning|everything|small model, lots of data (tens of thousands of rows+), one task, memory for ~16 B/param|large model, many tasks, little data|— (plain optimizer)|
|LoRA|low-rank deltas on chosen linears|default for most adaptation; many tasks share a base|no Linear layers in the path|`LoraConfig`|
|QLoRA|LoRA on a 4-bit base|base does not fit in bf16|MoE with fused experts; non-Linear-heavy models|`LoraConfig` + `BitsAndBytesConfig`|
|DoRA / AdaLoRA / LoRA+|variants of LoRA|chapters 12-14|serving needs hot-swap (DoRA/AdaLoRA must be merged)|`LoraConfig(use_dora=True)`, `AdaLoraConfig`|
|IA3|learned per-channel scaling vectors|very few parameters, little data|you need large behaviour change|`IA3Config`|
|prompt / prefix tuning|virtual tokens / per-layer prefixes|very little data, many tasks, frozen model|tasks far from pretraining; no merge possible|`PromptTuningConfig`, `PrefixTuningConfig`|
|adapters (bottleneck)|small MLPs inserted in blocks|research parity, some encoders|latency-sensitive serving (extra layers)|custom or other libraries|
|selective fine-tuning|chosen blocks unfrozen|no Linear targets; LSTM cells; normalisation drift|memory-bound large models|— (`requires_grad`)|
|head-only|the output head|new label space, features already good|task needs deeper change|— (`train_also=["classifier"]`)|
|projector-only|the modality bridge|vision/audio encoder + LM, new modality alignment|language behaviour must change|— / LoRA on projector|
|encoder-only / decoder-only side|one side of a seq2seq or multimodal model|the other side is already adequate|both sides mismatch the task|LoRA with target subsets|
|custom PEFT|whatever you implement|no framework supports the layer type|a supported method exists|`uadapt.lora` pattern (chapter 75)|

## 73.2 The decision, from constraints

```text
1. Output space new (new labels, new vocabulary)?  -> the HEAD must train (fully or with LoRA around it)
2. Memory (per GPU, Adam):
      full FT        ~16 bytes x params  (+ activations)
      frozen bf16    ~2 bytes x params   (+ adapter state + activations)
      frozen 4-bit   ~0.55 bytes x params (Linear layers only)
   pick the largest capacity that fits with ~30% headroom
3. Data rows: < 500 -> IA3 / prompt tuning / small-r LoRA; 500-50k -> LoRA; 50k+ and fits -> full FT is on the table
4. Tasks to serve: many -> adapters that hot-swap (LoRA/QLoRA/LoRA+), one per task
5. Architecture:
      MoE -> attention-only LoRA;  SSM -> big projections;  LSTM -> surrounding Linears or unfreeze the cell
      vision/audio encoder + LM -> projector + LM side first
6. Inference constraints: no extra latency allowed -> something mergeable (LoRA, not bottleneck adapters)
```

@file src/uadapt/strategy.py

```py
from uadapt.strategy import recommend
recommend(total_params=8_000_000_000, vram_gb=48, family=["text", "decoder-only"], task="sft",
          n_tasks=3, data_rows=20_000, lora_targets=224)["ranked"]
```

The tests assert the ranking is constraint-driven: full fine-tuning ranks first only when it fits and there is enough data; QLoRA appears when the bf16 base does not fit; MoE produces an attention-only option.

:::warn Scores rank options for your constraints; they are not a guarantee
Confirm any choice with a 20-step dry run (peak memory) and a small evaluation (does it learn the task?).
:::
