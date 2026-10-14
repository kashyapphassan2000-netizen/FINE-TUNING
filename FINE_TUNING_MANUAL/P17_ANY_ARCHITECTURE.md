# P17 — Any architecture: dense, MoE, MLA, hybrid/SSM, encoders, encoder-decoders, vision/audio, your own model, and models not released yet

**GOAL:** know, for any model you are handed, whether the standard path (P10) applies as-is, what changes, and how to adapt a model nobody has written a tutorial for.

## Step 0 — Ask the toolkit (works for any Hub id or local folder)

```bash
python -m peftkit.compat MODEL_ID
```

**YOU SHOULD SEE (real output for `Qwen/Qwen3-8B`, abridged):**

```json
{ "model": "Qwen/Qwen3-8B", "kind": "decoder-only", "model_type": "qwen3", "family": "dense", "attention": "gqa",
  "blockers": [], "warnings": [],
  "actions": ["1. peftkit.plan <model> -> pick the GPU", "2. write configs/models/<name>.yaml with the snippet below",
              "3. peftkit.diagnostics --config ... -> fix every FAIL", "4. 20-step dry run, then the real run"],
  "verdict": "SUPPORTED", "params_total": 8190735360 }
```

| Verdict | Means | Path |
|---|---|---|
| `SUPPORTED` | a causal LM the peftkit/TRL path handles | P10 as written |
| `SUPPORTED-ADVANCED` | supported with care (MoE, pre-quantized checkpoints, hybrid layers) | §2-§3 below |
| `OTHER-PIPELINE` | encoder, encoder-decoder, vision, audio: a different head/objective | §4 |
| `MANUAL-INTEGRATION` | custom code / unknown class | §5 (uadapt) |

For a deeper per-layer report (inputs, outputs, loss, LoRA targets, what not to adapt, risks) on any `nn.Module`:

```bash
python -m uadapt.report --model MODEL_ID --head causal-lm --vram-gb 80
python -m uadapt.report --custom path/to/my_model.py:MyModel --input "x=float:1,4,32" --vram-gb 24
```

## 1. Dense decoder (Llama, Qwen, Mistral, Gemma, Phi, …)

The default path. `target_modules: all-linear`; template from the tokenizer; `completion_only_loss: true` (P10). Gemma-style models with sliding-window layers: the planner's KV figure accounts for the windowed layers.

## 2. Mixture-of-Experts (Mixtral, Qwen3-MoE, DeepSeek-V3, Kimi-K2, gpt-oss)

| Fact | Consequence |
|---|---|
| **all experts** must sit in GPU memory | memory uses N_total (671B for DeepSeek-V3); time uses N_active (37.6B) |
| expert weights are often **fused 3-D tensors**, not `nn.Linear` | bitsandbytes cannot 4-bit them; QLoRA saves far less than on dense models (P03: DeepSeek-V3 QLoRA ≈ 1.36 TB) |
| `all-linear` targets every expert | billions of trainable parameters; use **attention-only** targets: `python -m peftkit.plan MODEL_ID --targets attention` |
| `prepare_model_for_kbit_training` upcasts non-4-bit tensors to fp32 | `quant.upcast_non_quantized: false` (planner flag `--no-upcast`) keeps them bf16 |
| routers / load balancing | do not adapt the router unless you know why; watch expert-load balance in logs if your framework reports it |
| scale | multi-node for 200B+; production MoE fine-tuning uses expert-parallel frameworks (Megatron-style, DeepSpeed-MoE): beyond this manual's single-node path |

## 3. MLA, hybrid/SSM, pre-quantized and long-context models

| Model type | What changes |
|---|---|
| **MLA** (DeepSeek-V2/V3, Kimi-K2) | LoRA targets are `q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj o_proj` (the planner lists them); the KV cache is tiny (T06) |
| **hybrid linear-attention / SSM** (Mamba-hybrids, Jamba-style, Qwen3-Next-style `layer_types`) | the planner reads `layer_types`; SSM/linear layers have a constant-size state (no growing KV). Kernels may need extra packages (`mamba-ssm`, `causal-conv1d`): check the model card |
| **pre-quantized checkpoint** (FP8, GPTQ, AWQ, MXFP4 in `quantization_config`) | PEFT usually needs a **bf16** copy: use the bf16 repo variant or dequantize first; the planner prints a NOTE |
| **MTP modules** (`num_nextn_predict_layers`) | dropped by most fine-tuning recipes; keep them in the checkpoint if you want speculative decoding later (T01 §2.5) |
| **long context** (extending 8k → 32k+) | set `rope_parameters` (YaRN/linear/dynamic), train on long examples, re-evaluate with needle tests (T06 §6); memory: logits and activations grow with sequence length |

## 4. Encoders, encoder-decoders, vision and audio models

| Family | Examples | Fine-tune for | How |
|---|---|---|---|
| encoder-only | BERT, RoBERTa, DeBERTa, ModernBERT | classification, NER, embeddings/retrieval | `AutoModelForSequenceClassification` / `TokenClassification` + PEFT `task_type="SEQ_CLS"`/`"TOKEN_CLS"`; train the new head fully (`modules_to_save`) |
| encoder-decoder | T5, Flan-T5, BART, Whisper | translation, summarisation, speech-to-text | `Seq2SeqTrainer`; LoRA on `q`, `v` (T5) or `q_proj`, `v_proj`; Whisper: the uadapt report above lists `fc1 fc2 k_proj out_proj q_proj v_proj` and the conv front-end |
| vision-language | Qwen2.5-VL / Qwen3-VL, LLaVA, Idefics | image + text → text | an `AutoProcessor` (images + text), a collator that keeps image tokens; LoRA on the **language** side first, the vision tower usually frozen; TRL SFT supports VLM datasets |
| vision / diffusion | ViT, CLIP, Stable Diffusion / FLUX | image classification, style/subject adaptation | LoRA on attention projections of the UNet/DiT (diffusers' LoRA training scripts); activations scale with **resolution**, so measure memory with a dry run |
| audio / TTS | Whisper, TTS models | domain vocabulary, new voices | as encoder-decoder; data = audio + transcripts; WER/CER metrics (`uadapt.metrics`) |

The toolkit's `uadapt` package has a task registry (19 tasks with schema, loss, collator and metrics: classification, NER, QA, seq2seq, CTC, contrastive, …) and a `UniversalPipeline`; worked chapters: `peft-universal/docs/master_src/76_encoder.md`, `77_encdec.md`, `78_vision.md`, `79_audio.md`, `80_multimodal.md`, `83_diffusion.md`.

## 5. Your OWN model (any `torch.nn.Module`, not in transformers)

**WHERE:** your PC (runs on CPU in seconds)
**GOAL:** LoRA on a model class only you have, with targets found by tracing (no layer names needed), trained, exported with an exact merge

```bash
PYTHONPATH=src python examples/custom_model/finetune_my_model.py
```

**YOU SHOULD SEE (real output):**

```text
inspect   : {'total': 9475, 'trainable': 9475, 'by_module_type': {'Linear': 8515, 'Conv1d': 832, 'LayerNorm': 128}}
classify  : ['custom']
LoRA targets found by tracing: ['mlp.0', 'mlp.2']
inject    : {'skipped': [], 'trainable': 1219, 'total': 10499, 'trainable_pct': 11.611}
before    : {'acc': 0.509765625}
gradients : {'ok': True}
after     : {'acc': 0.939453125}
export    : {'adapter': 'outputs/my_model_lora/adapter.pt', 'merged_layers': 2}
merge exact: True
serve     : CUSTOM ENGINEERING REQUIRED (no inference engine knows this class): torch inference behind FastAPI ...
```

**The pattern** (replace three things: the model, the batch, the loss):

```python
from uadapt.adapter import UniversalModelAdapter
ua = UniversalModelAdapter(my_model, sample_batch)          # sample_batch: dict of tensors your forward() takes
ua.inspect(); ua.classify()                                  # what is it, what does it take and return
ua.discover_trainable_modules()                              # LoRA targets by TRACING the forward pass
ua.build_training_pipeline(r=8, alpha=16, train_also=["head"])   # inject LoRA (Linear AND Conv), unfreeze a head
ua.validate_training(loss_fn, real_batch)                    # one backward: catches "nothing is learning" bugs
ua.train(loader, loss_fn, lr=2e-4, max_steps=1000)           # generic loop (AMP, accumulation, checkpoints)
ua.export("outputs/my_lora", merge=True)                     # adapter.pt + exactly merged weights
```

A `.pt` state dict of a **known** architecture: load it into the matching transformers class (`model.load_state_dict(torch.load(...))`), then `model.save_pretrained("my-model")` + the tokenizer: it is now a normal Hub-layout folder for P02-P14.

## 6. A model released next year (the method that does not age)

1. `python -m peftkit.compat NEW_MODEL --trust-remote-code` (only for repos you trust) → the verdict and blockers.
2. `python -m peftkit.plan NEW_MODEL` → exact counts on the meta device with the model's own code; if it cannot be built, the formula fallback is labelled. Compare with the model card.
3. Read `config.json` for anything new: attention type, `layer_types`, experts, quantization, rope.
4. `python -m uadapt.report --model NEW_MODEL` → targets, what not to adapt, risks.
5. Dry run 20 steps on the smallest GPU that fits → measured memory and speed.
6. Only then the real run. The formulas (P03, T06) do not change; only the inputs do.

## CHECKPOINT

- [ ] `peftkit.compat` verdict known for my model, and I know which section above applies.
- [ ] For MoE: attention-only targets considered, QLoRA memory checked with the planner.
- [ ] For my own model: `finetune_my_model.py` ran and I know the three things to replace.
