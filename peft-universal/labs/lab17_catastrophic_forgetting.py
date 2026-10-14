"""Lab 17: catastrophic forgetting, and how replay data reduces it.
'Pretrain' a tiny model on two skills, then fine-tune on ONE of them and measure the other."""
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
GENERAL = ["The pump pressure must stay below six bar during calibration.",
           "Reset the controller by holding the red button for five seconds.",
           "Service intervals are every six months for industrial pumps."]
DOMAIN = ["Warranty claims require the serial number and the purchase date.",
          "Claims without an invoice copy are rejected by the warranty team.",
          "A warranty claim is closed after ninety days without a reply."]


def batch(texts):
    enc = tok(texts, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def loss_on(model, texts):
    ids, attn = batch(texts)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, labels=ids.masked_fill(attn == 0, -100))
    return float(out.loss)


def train(model, texts, steps=120, lr=3e-3):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    ids, attn = batch(texts)
    for _ in range(steps):
        out = model(input_ids=ids, attention_mask=attn, labels=ids.masked_fill(attn == 0, -100))
        out.loss.backward(); opt.step(); opt.zero_grad()
    return model


base = AutoModelForCausalLM.from_pretrained(model_dir)
base = train(base, GENERAL + DOMAIN, steps=250)      # the "pretrained" model knows both
base.save_pretrained(tmp / "pre"); tok.save_pretrained(tmp / "pre")
g0, d0 = loss_on(base, GENERAL), loss_on(base, DOMAIN)
print(f"{'model':<34}{'general loss':>14}{'domain loss':>13}")
print(f"{'pretrained (knows both)':<34}{g0:>14.3f}{d0:>13.3f}")

for name, data, steps in [("fine-tuned on DOMAIN only", DOMAIN, 250),
                          ("fine-tuned on DOMAIN + 25% replay", DOMAIN * 3 + GENERAL, 250)]:
    m = train(AutoModelForCausalLM.from_pretrained(tmp / "pre"), data, steps=steps)
    g, d = loss_on(m, GENERAL), loss_on(m, DOMAIN)
    print(f"{name:<34}{g:>14.3f}{d:>13.3f}   general {'WORSE' if g > g0 else 'ok'} "
          f"({g - g0:+.3f}), domain {d - d0:+.3f}")
print("\nThis is full fine-tuning, the worst case. PEFT forgets less (the base weights are frozen),")
print("but it still happens: always measure a GENERAL suite before and after, not only your domain.")
print("Mitigations: mix 5-30% general/replay data, fewer epochs, lower lr, lower rank, keep the adapter separate.")
