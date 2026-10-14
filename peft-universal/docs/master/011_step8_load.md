# 11. Step 8 — Load the model on the GPU and prove it runs

```gpu
huggingface-cli login                      # or HF_TOKEN in .env (gated models need the accepted licence)
python - <<'PY'
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "<org/model>"; t0 = time.time()
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto",
                                             attn_implementation="sdpa")
print("load seconds:", round(time.time() - t0, 1))
print("device map:", getattr(model, "hf_device_map", "single GPU"))
print("params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
print("peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
enc = tok.apply_chat_template([{"role": "user", "content": "Say OK."}], add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=8)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

## 11.1 What to check, in order

|Check|Healthy|If not|
|---|---|---|
|download time|matches chapter on load maths (size ÷ bandwidth)|pre-download to a persistent volume|
|peak VRAM after load|close to `peftkit.plan`'s weights figure|wrong dtype (fp32 doubles it) or a device map spilling to CPU|
|`hf_device_map`|one GPU, or the tensor-parallel layout you intended|`device_map="auto"` silently offloading to CPU/disk is why "it is so slow"|
|a generated sentence|coherent for an instruct model|wrong tokenizer/template, or a base (non-instruct) checkpoint|
|`nvidia-smi`|memory held by your process only|another process sharing the card|

## 11.2 Quantized loading (when it does not fit)

```gpu
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained("<org/model>", quantization_config=q, device_map="auto")
print(round(torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")
PY
```

Remember what 4-bit does **not** cover: convolutions, fused MoE experts, embeddings and norms. On an MoE model the saving can be far smaller than expected (the MoE chapter measures this).

:::check
- The model loads, generates a sentence, and the peak VRAM matches the plan within ~10%.
:::
