# 34. Mapping ANY architecture (including ones that do not exist yet)

:::why The problem
Llama, Qwen, Mistral, Gemma, Phi, Mixtral, gpt-oss, DeepSeek-V3, Kimi-K2, Qwen3-Next, Falcon-H1, Llama-4 all differ: different layer names, different attention, dense or MoE, some with custom code. A fine-tuning template that hard-codes `["q_proj","v_proj"]` breaks on half of them. This chapter is the procedure that never breaks: **ask the model, never assume**.
:::

## 34.1 First: which models can you fine-tune at all?

|Model|Weights public?|Can you LoRA it yourself?|
|---|---|---|
|Llama, Qwen, Mistral/Mixtral, Gemma, Phi, DeepSeek, Kimi, gpt-oss, Falcon, GLM, Granite, ...|yes (licences differ)|yes, this template|
|GPT-4 / GPT-5 class (OpenAI), Claude (Anthropic), Gemini (Google)|**no**|**No. You cannot download the weights.** Their architectures are not public either|

For closed models your options are: prompt engineering, RAG, or the vendor's own fine-tuning service **if** it offers one for that model (check the vendor's current documentation; availability and supported models change). Nothing in this repo applies to closed weights. Anyone claiming to "fine-tune GPT-4 weights locally" is wrong.

The rest of this chapter is about open-weight models.

## 34.2 The five things that differ

|Axis|Options|Where you see it in `config.json`|
|---|---|---|
|dense vs MoE|one MLP per layer vs many experts + router|`n_routed_experts` / `num_local_experts` / `num_experts`|
|attention|MHA, GQA, MQA, MLA, sliding-window mixes|`num_key_value_heads`, `kv_lora_rank`, `sliding_window`, `layer_types`|
|layer mix|all attention, or attention + linear-attention/SSM (hybrid)|`layer_types`|
|naming|`q_proj` vs `q_a_proj`/`q_b_proj` vs fused `qkv_proj` vs `in_proj_qkvz`|the module names|
|packaging|plain transformers code vs `trust_remote_code` custom code; text vs multimodal; pre-quantized|`architectures`, `auto_map`, `text_config`, `quantization_config`|

## 34.3 The verified matrix (generated from transformers 5.17 on this machine)

@lab arch_matrix

Read it as: what you would put in `method.target_modules` for attention-only adaptation (column 4), what `all-linear` adds (column 5), and which tensors are fused 3-D expert weights (column 6).

:::warn Three things this matrix teaches
- **MLA models (DeepSeek-V3, Kimi-K2) have no `q_proj`/`k_proj`/`v_proj` at all.** Copying a Llama target list gives you zero adapters or an error.
- **Hybrid models (Qwen3-Next, Falcon-H1, Kimi-Linear)** put linear-attention/SSM projections (`in_proj_qkvz`, `in_proj`, `out_proj`) in the layer list. Adapting them is possible but under-studied; start with the full-attention layers and the MLP.
- The fused-tensor detector flags any 3-D parameter, which also catches SSM convolution weights (`weight` in the Falcon-H1 row). Look at the names before targeting.
:::

### Sizes, KV cost and fused tensors for the same families

@lab arch_family_map

Read the KV column against chapter 36.5: MoE with MLA (deepseek_v3) caches far less per token than a dense MHA model, which is why those giant models are servable at long context.

:::warn Limit of automatic detection
Detection reads the config. `falcon_h1` shows as dense/gqa above because its config carries no `layer_types`, although it is a hybrid attention+SSM model. Treat the matrix as a fast first pass, then confirm with the model card and `peftkit.inspect_model`.
:::

## 34.4 The universal procedure for ANY model, today or in two years

```any Step 1: what is it? (downloads config.json only, never the weights)
python -m peftkit.plan <org>/<model> --seq-len 4096
```

It prints family, attention type, layer counts, exact parameter counts, the LoRA target names, fused expert tensors, warnings about pre-quantized checkpoints, memory per method and GPU counts. If it says `formula fallback`, the architecture could not be built locally, which means one of:

- the repo needs `--trust-remote-code` (custom modelling code): **read the code in the repo first**, then re-run with the flag;
- your transformers version is older than the model: upgrade transformers, or use the vendor's fork;
- the repo is multimodal with an unusual wrapper: point the planner at the `text_config`.

```any Step 2: confirm the module names on the real object (meta device, no weights downloaded)
python -m peftkit.inspect_model <org>/<model> [--trust-remote-code]
```

```any Step 3: write the model YAML with what you just saw
# configs/models/<model>.yaml
model: {name_or_path: <org>/<model>, revision: <commit sha>, max_seq_length: <p95 of your tokens>}
method: {target_modules: [<names from step 2>]}     # or all-linear for dense models
train: {output_dir: outputs/<name>, per_device_batch_size: <from dry run>, grad_accum: <...>}
```

```any Step 4: 20-step dry run, then read what actually got adapted
python -m peftkit.train --config configs/base.yaml configs/methods/lora.yaml configs/models/<model>.yaml \
   --set train.max_steps=20
python -c "import json;m=json.load(open('outputs/<name>/run_meta.json'));print(m['trainable_params'], m['peak_vram_gb'])"
```

```any Step 5: can vLLM serve it? Check the architecture is registered, then check your adapter format
python -c "from vllm.model_executor.models.registry import ModelRegistry; print('<Arch>ForCausalLM' in ModelRegistry.get_supported_archs())"
python -m peftkit.serve_check outputs/<name>/adapter
```

vLLM 0.29.0 registers 200+ causal-LM architectures, and the families checked for this guide (Llama, Mixtral, Qwen3-MoE, DeepSeek-V3, gpt-oss, Qwen3-Next, Gemma-3) all declare LoRA support. Kimi-K2 ships as a DeepSeek-V3-style architecture with custom code: check the `architectures` field of its `config.json` and vLLM's release notes for your version.

## 34.5 Checklist for a brand-new architecture

:::check Before you spend money on an unknown model
- `config.json` read; `architectures`, `model_type`, `auto_map` noted.
- `peftkit.plan` gives exact counts (not the fallback).
- If `trust_remote_code` is needed, I have read that code (it executes on my machine).
- Target module names come from `inspect_model`, not from a blog post.
- Tokenizer has a chat template (`print(tok.chat_template)`), or I chose one deliberately.
- A 20-step dry run shows a non-zero `trainable_params` and a loss that moves.
- `serve_check` says adapter or merge, and the architecture is in vLLM's registry.
- Licence read: some open-weight models restrict commercial use or model-output use.
:::

## 34.6 When this template is not the right tool

Be honest with yourself early:

- The model needs multi-node training (chapter 36's ladder says > 8 GPUs): use a framework built for it (DeepSpeed, Megatron-LM / NeMo, torchtitan, ms-swift, LLaMA-Factory) and treat this repo as the data + evaluation + serving half.
- The model is multimodal (image/audio in): you need its processor and a collator for those inputs. The PEFT ideas carry over; this code does not.
- The checkpoint is FP8/MXFP4 and your framework cannot train it: convert to bf16 first (chapter 36.9) or pick another checkpoint.
- The model is closed-weight: nothing here applies (36.1).
