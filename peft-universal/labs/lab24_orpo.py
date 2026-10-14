"""Lab 24: ORPO - SFT and preference alignment in ONE stage, no reference model, no reward model.

ORPO (Hong et al., 2024) loss per pair:
    L = L_SFT(chosen)  +  lambda * L_OR
    L_OR = -log sigmoid( log odds(chosen) - log odds(rejected) ),   odds(y) = P(y|x) / (1 - P(y|x))
P(y|x) is the length-normalised sequence likelihood. The SFT term teaches the chosen answer; the odds-ratio
term pushes the rejected one down. No frozen reference copy (DPO needs one) -> about half the memory of DPO.
In TRL 1.13 ORPO lives in trl.experimental.orpo (it was moved out of the top-level package).
"""
import json, logging, sys, tempfile, warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl.experimental.orpo import ORPOConfig, ORPOTrainer  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
tok = AutoTokenizer.from_pretrained(model_dir)
PAIRS = [("What is the maximum pressure?", "6 bar.", "I think maybe around twenty bar, not sure."),
         ("How do I reset?", "Hold the red button five seconds.", "Try turning it off and on a few times."),
         ("What do warranty claims need?", "Serial number and purchase date.", "Just send an email."),
         ("When is service due?", "Every six months.", "Whenever it feels broken.")]
rows = [{"prompt": [{"role": "user", "content": q}],
         "chosen": [{"role": "assistant", "content": c}],
         "rejected": [{"role": "assistant", "content": r}]} for q, c, r in PAIRS] * 4
(tmp / "pref.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
ds = load_dataset("json", data_files={"train": str(tmp / "pref.jsonl")})["train"]

args = ORPOConfig(output_dir=str(tmp / "orpo"), num_train_epochs=20, per_device_train_batch_size=4, disable_tqdm=True,
                  learning_rate=5e-3, beta=0.1, max_length=128, logging_steps=10, save_strategy="no",
                  report_to=[], bf16=False, use_cpu=True, seed=0)
trainer = ORPOTrainer(model=AutoModelForCausalLM.from_pretrained(model_dir), args=args, train_dataset=ds,
                      processing_class=tok,
                      peft_config=LoraConfig(r=8, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
keys = sorted({k for h in trainer.state.log_history for k in h})
want = [k for k in ("loss", "nll_loss", "log_odds_ratio", "rewards/accuracies", "rewards/margins") if k in keys]
print(f"{'step':>6}" + "".join(f"{k:>20}" for k in want))
for h in trainer.state.log_history:
    if "loss" in h and "step" in h:
        print(f"{h['step']:>6}" + "".join(f"{h.get(k, float('nan')):>20.4f}" for k in want))
print("\nlogged keys this TRL version produces:", [k for k in keys if "/" in k or "odds" in k or "nll" in k])
print("""
READING
- nll_loss is the SFT part (falls as the chosen answers are learned).
- log_odds_ratio / rewards/margins is the preference part (chosen pulled above rejected).
- ORPO starts from the BASE or an instruct model directly: it does SFT and alignment together.
- lr 5e-3 is ONLY for this random toy model; real models use 5e-6 (full) to 5e-5..2e-4 (LoRA).
- beta here is ORPO's lambda (weight of the odds-ratio term); 0.1 is the paper's default range (0.1-0.5).
- Use ORPO when you have pairs from the start and limited GPU memory; use SFT -> DPO when you already have a
  good SFT model and want a separate, measurable alignment step.""")
