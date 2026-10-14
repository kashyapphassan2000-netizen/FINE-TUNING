"""Lab 23: preference optimization with DPO, after SFT, on the tiny model (CPU)."""
import json, logging, sys, tempfile
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

logging.disable(logging.WARNING)
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

args = DPOConfig(output_dir=str(tmp / "dpo"), num_train_epochs=8, per_device_train_batch_size=4,
                 learning_rate=5e-4, beta=0.1, max_length=128, logging_steps=4, save_strategy="no",
                 report_to=[], bf16=False, use_cpu=True, seed=0)
trainer = DPOTrainer(model=AutoModelForCausalLM.from_pretrained(model_dir), args=args, train_dataset=ds,
                     processing_class=tok, peft_config=LoraConfig(r=8, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
hist = [h for h in trainer.state.log_history if "rewards/accuracies" in h]
print(f"{'step':>6}{'loss':>10}{'reward acc':>12}{'margin':>10}{'chosen':>10}{'rejected':>10}")
for h in hist:
    print(f"{h['step']:>6}{h['loss']:>10.4f}{h['rewards/accuracies']:>12.3f}{h['rewards/margins']:>10.4f}"
          f"{h['rewards/chosen']:>10.4f}{h['rewards/rejected']:>10.4f}")
print("\nDPO needs no reward model: it raises the log-probability of the chosen answer and lowers the rejected one,")
print("relative to a frozen reference copy of the same model. beta controls how far it may drift from the reference.")
print("Read 'reward accuracy' (should rise toward 1.0) and 'margin' (should grow). Loss alone says little.")
print("Order: SFT first, then DPO on preference pairs built in Volume 1, chapter 16. Evaluate both stages separately.")
print(f"(trl DPOConfig in this version has no max_prompt_length field; torch.cuda.is_available() = {torch.cuda.is_available()})")
