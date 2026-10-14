# 33. Model compatibility: "can I fine-tune ANY model?"

The honest answer: **any decoder-only causal LM whose code your transformers version can build.** Everything else needs a different pipeline, and pretending otherwise is how people lose a week.

## 33.1 The compatibility matrix

|Model family|Example|This template|What you need instead|
|---|---|---|---|
|Decoder-only causal LM|Llama, Qwen, Mistral, Gemma, Phi, SmolLM|**Yes**|-|
|MoE decoder-only|Mixtral, Qwen3-MoE, DeepSeek-V3, gpt-oss|**Yes, advanced**|choose targets deliberately; see chapter 36.8|
|Encoder-only|BERT, RoBERTa, DeBERTa|No|`AutoModelForSequenceClassification` + a classification trainer; PEFT supports it with `task_type="SEQ_CLS"`|
|Encoder-decoder|T5, FLAN-T5, BART|No|`AutoModelForSeq2SeqLM` + `Seq2SeqTrainer`; `task_type="SEQ_2_SEQ_LM"`|
|Vision-language / multimodal|Llava, Qwen-VL, Llama-4 multimodal|No|a processor (not just a tokenizer) and an image-aware collator|
|Speech|Whisper|No|audio pipeline|
|Custom / brand-new architecture|remote-code repos|Manual|read the repo's `.py`, use `--trust-remote-code`, verify layer names by hand|
|Closed models|GPT, Claude, Gemini|Impossible|the weights are not public. Use the vendor's own tuning service if it offers one, or an open model|

## 33.2 The automated check

@file src/peftkit/compat.py

```any
python -m peftkit.compat meta-llama/Llama-3.1-8B-Instruct
python -m peftkit.compat <org>/<some-new-model> --trust-remote-code   # only after reading their code
```

It reports: kind (decoder-only / encoder-decoder / multimodal / unknown), family (dense/MoE), attention type, exact parameter counts, **the real target-module names**, fused expert tensors, tokenizer facts (chat template, generation markers, pad/eos), a verdict, blockers, warnings, and a ready-to-paste config snippet. It exits non-zero when the verdict is not supported, so CI can use it.

## 33.3 The seven things that must line up

1. **Architecture** is decoder-only causal LM.
2. **transformers can build it** (or you accept remote code).
3. **Tokenizer** loads, has a pad token (or one is assigned), and a chat template.
4. **Chat template** has `{% generation %}` markers if you want assistant-only loss.
5. **Target modules** exist under those names in this architecture.
6. **Quantization**: if the checkpoint is pre-quantized (FP8/MXFP4/GPTQ/AWQ), make a bf16 copy for PEFT training.
7. **Serving**: vLLM supports the architecture, and the adapter format is servable (LoRA, not DoRA/AdaLoRA) — `peftkit.serve_check`.

## 33.4 A brand-new model, released tomorrow

```text procedure
1. peftkit.compat <model>            -> kind, verdict, target names, tokenizer facts
2. peftkit.plan <model>              -> memory, GPU count, KV cache, cost inputs
3. write configs/models/<name>.yaml from the printed snippet (pin `revision`)
4. peftkit.diagnostics --config ...  -> fix every FAIL
5. 20-step dry run                   -> peak VRAM, tokens/sec
6. full run -> evaluate vs baseline -> merge -> serve_check -> serve
```

If step 1 says MANUAL-INTEGRATION: read the repo's modeling file, find the `nn.Linear` names inside the decoder layer, set `target_modules` explicitly, and verify `trainable params > 0` before training.

:::check
- `peftkit.compat` exits 0 for my model.
- I read its warnings, not just the verdict.
- My config snippet pins a revision.
:::
