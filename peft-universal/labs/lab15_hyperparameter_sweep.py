"""Lab 15: a real hyperparameter sweep on the tiny model, through peftkit.experiments."""
import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402

from peftkit.experiments import as_markdown, run_grid  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp())
model_dir = build_tiny(tmp / "tiny")
QA = [("What is the maximum pressure?", "6 bar."), ("How do I reset?", "Hold the red button five seconds."),
      ("What do warranty claims need?", "Serial number and purchase date."),
      ("When is service due?", "Every six months.")]
rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in QA]
(tmp / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows * 12))
(tmp / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows * 2))

base = dict(model={"name_or_path": str(model_dir), "max_seq_length": 128},
            data={"train_file": str(tmp / "train.jsonl"), "eval_file": str(tmp / "eval.jsonl")},
            train={"epochs": 6, "per_device_batch_size": 8, "grad_accum": 1, "logging_steps": 6,
                   "eval_steps": 18, "save_steps": 10000, "seed": 0})
print("A) learning rate (r=8 fixed). Everything else identical.")
lr_rows = run_grid([str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml")],
                   {"train.lr": [2e-5, 2e-4, 2e-3, 5e-2], "method.r": [8]}, str(tmp / "lr"), extra=base)
print(as_markdown(lr_rows, ["lr", "trainable", "train_loss", "eval_loss", "eval_token_acc", "seconds"]))
print("\nB) LoRA rank (lr=2e-3 fixed).")
r_rows = run_grid([str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml")],
                  {"method.r": [2, 8, 32], "train.lr": [2e-3]}, str(tmp / "r"), extra=base)
print(as_markdown(r_rows, ["r", "trainable", "train_loss", "eval_loss", "eval_token_acc", "adapter_mb", "seconds"]))
print("\nRead it as: too low an lr barely moves; too high can diverge; rank buys capacity, and capacity")
print("costs memory and adapter size. Your own numbers will differ - that is why you sweep.")
