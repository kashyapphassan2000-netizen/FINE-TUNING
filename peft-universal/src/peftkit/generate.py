"""Quick sanity generation with plain transformers (base, adapter, or merged). Not for serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(path: str):
    p = Path(path)
    cuda = torch.cuda.is_available()
    kw = dict(dtype=torch.bfloat16 if cuda else torch.float32, device_map="auto" if cuda else None)
    if (p / "adapter_config.json").exists():
        from peft import PeftModel

        base = json.loads((p / "adapter_config.json").read_text())["base_model_name_or_path"]
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, **kw), path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    return model.eval(), AutoTokenizer.from_pretrained(path)


def chat(model, tok, prompt: str, max_new_tokens: int = 128, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = ap.parse_args(argv)
    m, t = load(a.model)
    print(chat(m, t, a.prompt, a.max_new_tokens))


if __name__ == "__main__":
    main()
