"""Merge a LoRA adapter into its base model and (optionally) publish it - standard libraries only.
Equivalent of `peftkit.merge` + `peftkit.publish`, including the check that the merge is exact.

python examples/standard/merge_publish_raw.py --base outputs/tiny --adapter outputs/raw_lesson/adapter \
    --out outputs/raw_lesson/merged [--hub your-user/your-repo --private]
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hub", default=None)
ap.add_argument("--private", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.adapter)
base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
model = PeftModel.from_pretrained(base, a.adapter).eval()

# prove the merge is exact: logits before and after on the same input
# transformers 5: apply_chat_template(..., return_tensors="pt") returns a BatchEncoding (dict-like),
# not a tensor. Ask for the dict explicitly and take input_ids - passing the object itself fails.
ids = tok.apply_chat_template([{"role": "user", "content": "What is the maximum pressure?"}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
with torch.no_grad():
    before = model(input_ids=ids).logits
    merged = model.merge_and_unload()        # folds B·A·(alpha/r) into W, removes the LoRA modules
    after = merged(input_ids=ids).logits
print("max |logit difference| after merge:", float((before - after).abs().max()))

merged.save_pretrained(a.out)
tok.save_pretrained(a.out)
print("saved merged model to", a.out)
if a.hub:                                     # needs HF_TOKEN (write) in the environment or `hf auth login`
    merged.push_to_hub(a.hub, private=a.private)
    tok.push_to_hub(a.hub, private=a.private)
    print("published to https://huggingface.co/" + a.hub)
