# 129. The unknown-model challenge, version 2

The review's point 17: make the test harder, and make it a **16-point determination** you produce **without changing the engine**.

## 129.1 The rules

You are given a directory and a one-line business request. Before writing any training code, produce this table. Then run `python -m uadapt.report ...` and compare — every difference is a lesson.

|#|Determine|Tool / evidence|
|---|---|---|
|1|modality|forward signature, config keys, processor files (ch 83-84)|
|2|architecture family|decision tree (ch 84)|
|17|support level 1-5|chapter 119|
|18|task (training objective)|`tasks.resolve` + the business request (ch 121)|
|19|processor / feature extractor|files in the repo; the model's own class|
|20|tokenizer / vocabulary / special tokens|tokenizer files; chat template presence|
|21|input structure|`inspect.signature(model.forward)`|
|22|output structure|one forward pass; output keys and shapes|
|23|loss|source scan + does it return `loss`? (ch 86)|
|24|trainable modules|`targets.discover` roles (ch 88)|
|25|PEFT compatibility|Linear/Conv present? packed params? fused experts? (ch 89)|
|26|VRAM for training and serving|`peftkit.plan`, `peftkit.kv`, dry run (ch 106-107)|
|27|dataset format|the task's `data_schema` (ch 121)|
|28|collator|the task's collator rules (ch 101)|
|29|evaluation suites + metrics|`TASK_METRICS` + your own eval set (ch 103, 107)|
|30|checkpoint format and serving stack|merge or adapter; engine by architecture (ch 104)|

## 129.2 Six challenges, harder than chapter 115's

1. **A 1.5B "reasoning" decoder** whose config has `head_dim` absent, `tie_word_embeddings: true`, and a chat template with tool-call tags. Business request: extract structured JSON from support emails. *(Which task? Is the chat template usable? What breaks if `tie_word_embeddings` is true and you merge?)*
2. **An encoder with a projection head** (`*Model` + `sentence_transformers` config). Request: retrieval over 2M internal documents on one 24 GB GPU. *(Task, loss, batch-size pressure, index and serving; how do you measure success?)*
3. **A speech model with `codec` in its config and no `lm_head`.** Request: clone a support agent's voice for IVR. *(Which task is this? What consent and legal record is required? What loss can you honestly ship?)*
4. **A DiT-style image model with a `vae/` folder and `scheduler_config.json`.** Request: product photos in a house style from 12 reference images. *(Which variant: LoRA, DreamBooth, textual inversion? What is the evaluation set and the regression check?)*
5. **A multimodal model that accepts 1-8 images per sample** with tiling, `trust_remote_code`, and a custom `processing_x.py`. Request: read equipment labels from technician photos. *(Placeholder accounting, KV budget with tiling, what to adapt first, per-segment evaluation.)*
6. **A `nn.Module` with no config, mixing a learned token-mixing `nn.Parameter`, grouped convolutions and `nn.MultiheadAttention`.** Request: fine-tune it on 5,000 conversations and serve it to 200 users. *(Which parts can be LoRA'd at all? What must be re-implemented? Which serving stack, and what capacity?)*

## 129.3 How to grade yourself

- **Pass**: all 16 rows answered with evidence, and your plan survives the one-batch smoke test.
- **Strong**: you also predicted the traps (packed parameters, tied embeddings, placeholder counts, CTC frame limits, licence conflicts) before hitting them.
- **Expert**: you produced the capacity and cost numbers, the release-gate criteria and the honest "what I am NOT claiming" list before training a single step.
