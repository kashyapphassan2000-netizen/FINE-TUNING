"""Lab 22: continued pretraining (raw text, every token trained) before SFT (chat, answers only)."""
import sys, tempfile
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
tok = AutoTokenizer.from_pretrained(model_dir)
CORPUS = ["Impeller cavitation appears as pitting on the vane surface of industrial pumps.",
          "Mechanical seal leakage above two drops per minute requires immediate replacement.",
          "Cavitation is reduced by raising suction pressure or lowering the pump speed."]
HELD = ["Cavitation damages the impeller vane surface of industrial pumps."]


def lm_loss(model, texts):
    enc = tok(texts, return_tensors="pt", padding=True)
    labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    with torch.no_grad():
        return float(model(**enc, labels=labels).loss)


def train_lm(model, texts, steps=200, lr=3e-3):
    enc = tok(texts, return_tensors="pt", padding=True)
    labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        out = model(**enc, labels=labels)
        out.loss.backward(); opt.step(); opt.zero_grad()
    return model


m = AutoModelForCausalLM.from_pretrained(model_dir)
before = lm_loss(m, HELD)
m = train_lm(m, CORPUS)
after = lm_loss(m, HELD)
print(f"held-out domain sentence, loss before continued pretraining: {before:.3f}")
print(f"held-out domain sentence, loss after  continued pretraining: {after:.3f}  ({after - before:+.3f})")
print("\nContinued pretraining (CPT) = keep doing next-token prediction on RAW domain text. No chat template,")
print("no loss masking, every token counts. It teaches vocabulary and domain patterns, not instruction following.")
print("\nWhen to use which:")
print("  new jargon / new language / a whole technical field  -> CPT first (needs a lot of text), then SFT")
print("  behaviour, format, tone, task skill                   -> SFT only (this is most projects)")
print("  facts that change                                     -> retrieval, not training (lab 21)")
print("Order of a full post-training pipeline: CPT -> SFT -> preference optimization (lab 23) -> evaluation.")
print("CPT with PEFT usually needs a higher rank than SFT, and it can cause forgetting (lab 17): keep a general suite.")
