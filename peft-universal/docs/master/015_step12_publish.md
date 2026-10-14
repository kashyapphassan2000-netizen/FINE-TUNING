# 15. Step 12 — Merge the best checkpoint and publish to Hugging Face

```gpu
python -m peftkit.merge --adapter outputs/<run>/checkpoint-<best> --out outputs/<run>/merged --base <org/model>
python -m peftkit.serve_check outputs/<run>/adapter
python -m peftkit.publish --src outputs/<run>/merged --hub <you>/<repo> --private
```

|Step|What it guarantees|
|---|---|
|merge onto the exact base + revision|the merged model is the model you evaluated (max logit difference ~1e-6)|
|`serve_check`|the adapter format is servable (DoRA/AdaLoRA must be merged, not hot-swapped)|
|publish `--private` first|you can test the download path before anyone sees it|

## 15.1 What to upload besides weights

- the **model card**: base model + revision, method and hyperparameters, dataset description and lineage, evaluation numbers **with the suites named**, known limitations, licence and its inherited duties;
- the **tokenizer and processor** exactly as trained (a mismatch here is the classic serving bug);
- optionally the **adapter** as a separate repo, if you serve multi-LoRA.

## 15.2 Verify the round trip before you close the session

```gpu
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("<you>/<repo>"); t = AutoTokenizer.from_pretrained("<you>/<repo>")
enc = t.apply_chat_template([{"role":"user","content":"What is the maximum pressure?"}],
                            add_generation_prompt=True, return_tensors="pt", return_dict=True)
print(t.decode(m.generate(**enc, max_new_tokens=32)[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

Then **delete the training VM**. The weights are on the Hub; the machine is a cost.

:::check
- Download-and-generate from the Hub works from a clean machine.
- Model card records the evaluation numbers and the licence duties.
:::
