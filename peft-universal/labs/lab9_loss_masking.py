"""Lab 9: what breaks without loss masking. Same data, same steps, only `assistant_only_loss` differs.
Data has a long fixed system prompt (like most enterprise data) and short answers."""
import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import torch  # noqa: E402
from tiny import build_tiny  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from peftkit.config import load_config  # noqa: E402
from peftkit.train import run  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp())
model_dir = build_tiny(tmp / "tiny")
SYSTEM = ("You are the calibration assistant for the pump controller. Always follow the warranty rules "
          "and the reset procedure. The pump pressure must stay below six bar during calibration. " * 2)
QA = [("What is the maximum pressure?", "6 bar."), ("How do I reset?", "Hold red button five seconds."),
      ("Warranty needs?", "Serial number and purchase date.")]
rows = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                      {"role": "assistant", "content": a}]} for q, a in QA] * 16
(tmp / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
tok = AutoTokenizer.from_pretrained(model_dir)


def answer_nll(model) -> float:
    """Mean negative log-likelihood of the ANSWER tokens only (what we actually care about)."""
    total, count = 0.0, 0
    for q, a in QA:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]
        enc = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                                      return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"]).logits[0, :-1]
        target = enc["input_ids"][0, 1:]
        mask = enc["assistant_masks"][0, 1:].bool()
        nll = torch.nn.functional.cross_entropy(logits[mask], target[mask], reduction="sum")
        total, count = total + nll.item(), count + int(mask.sum())
    return total / count


results = {}
for masked in (False, True):
    out = tmp / f"run_{masked}"
    cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides={
        "model": {"name_or_path": str(model_dir), "max_seq_length": 256},
        "method": {"r": 8},
        "data": {"train_file": str(tmp / "train.jsonl"), "eval_file": None, "assistant_only_loss": masked},
        "train": {"output_dir": str(out), "epochs": 2, "per_device_batch_size": 8, "grad_accum": 1, "lr": 5e-3,
                  "logging_steps": 3, "save_steps": 1000, "seed": 0},
    })
    meta = run(cfg)
    from peft import PeftModel

    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), out / "adapter").eval()
    results[masked] = (meta["train_loss"], answer_nll(m))
base = AutoModelForCausalLM.from_pretrained(model_dir).eval()
print(f"untrained model          answer NLL = {answer_nll(base):.3f}")
for masked, (tl, nll) in results.items():
    label = "assistant-only loss" if masked else "loss on ALL tokens "
    print(f"{label}  reported train loss = {tl:.3f}   answer NLL = {nll:.3f}")
print("Reported train losses are NOT comparable: they average over different token sets.")
