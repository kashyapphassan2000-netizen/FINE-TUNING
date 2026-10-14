"""The SAME fine-tuning run as `python -m peftkit.train`, written with standard libraries only
(transformers + peft + trl + datasets). No peftkit import anywhere. If this repository ever disappears,
this file is still a complete, working recipe.

python examples/standard/train_sft_raw.py --model outputs/tiny --train data/lesson/train.jsonl \
    --eval data/lesson/eval.jsonl --out outputs/raw_lesson --epochs 20 --lr 5e-3
Add --qlora on an NVIDIA GPU to load the base model in 4-bit.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--eval", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--max-length", type=int, default=1024)
ap.add_argument("--qlora", action="store_true")
a = ap.parse_args()

# 1. tokenizer: the model's OWN chat template is used for formatting and for the assistant-only mask
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 2. model: bf16 on GPU, fp32 on CPU; 4-bit NF4 for QLoRA (Linear layers only)
gpu = torch.cuda.is_available()
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16) if a.qlora else None
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if gpu else torch.float32,
                                             quantization_config=quant, attn_implementation="sdpa")

# 3. LoRA: which layers, what rank, what scale. "all-linear" = every Linear except the output head
lora = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules="all-linear",
                  task_type="CAUSAL_LM")

# 4. data: JSONL with {"messages": [...]} rows
files = {"train": a.train} | ({"eval": a.eval} if a.eval else {})
ds = load_dataset("json", data_files=files)

# 5. training arguments: these are the knobs peftkit's YAML sets for you
args = SFTConfig(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
    per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
    lr_scheduler_type="cosine", warmup_steps=0.03,      # transformers 5: a float here is a ratio
    max_length=a.max_length, assistant_only_loss=True,  # loss on the assistant turns only
    bf16=gpu, gradient_checkpointing=gpu, logging_steps=20,
    eval_strategy="epoch" if a.eval else "no", save_strategy="epoch", save_total_limit=2,
    report_to=[], seed=0,
)

# 6. train: SFTTrainer applies the chat template, the mask, and attaches LoRA from peft_config
trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds.get("eval"),
                     processing_class=tok, peft_config=lora)
trainer.model.print_trainable_parameters()
result = trainer.train()
trainer.save_model(f"{a.out}/adapter")          # adapter weights + adapter_config.json
tok.save_pretrained(f"{a.out}/adapter")
print({"train_loss": round(result.training_loss, 4), "steps": result.global_step, "saved": f"{a.out}/adapter"})
