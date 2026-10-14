"""Lab 16: overfitting vs underfitting, seen in the loss curves of real runs."""
import json, logging, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from peftkit.config import load_config  # noqa: E402
from peftkit.train import run  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
TRAIN = [("What is the maximum pressure?", "6 bar."), ("How do I reset?", "Hold the red button five seconds."),
         ("What do warranty claims need?", "Serial number and purchase date."),
         ("When is service due?", "Every six months.")]
HELD = [("What pressure is allowed?", "6 bar."), ("How is the controller reset?", "Hold the red button five seconds."),
        ("What is needed for a warranty claim?", "Serial number and purchase date."),
        ("How often is service needed?", "Every six months.")]
row = lambda qa: [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in qa]
(tmp / "train.jsonl").write_text("\n".join(json.dumps(r) for r in row(TRAIN) * 2))   # only 8 examples
(tmp / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in row(HELD)))
cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
    model={"name_or_path": str(model_dir), "max_seq_length": 128}, method={"r": 16},
    data={"train_file": str(tmp / "train.jsonl"), "eval_file": str(tmp / "eval.jsonl")},
    train={"output_dir": str(tmp / "run"), "epochs": 60, "per_device_batch_size": 8, "grad_accum": 1,
           "lr": 5e-3, "logging_steps": 10, "eval_steps": 10, "save_steps": 100000, "seed": 0}))
meta = run(cfg)
hist = meta["log_history"]
train = {h["step"]: h["loss"] for h in hist if "loss" in h}
ev = {h["step"]: (h["eval_loss"], h.get("eval_mean_token_accuracy")) for h in hist if "eval_loss" in h}
print("8 training examples, 4 unseen eval examples, 60 epochs. Watch the two columns separate.")
print(f"{'step':>6}{'train loss':>12}{'eval loss':>11}{'eval token acc':>16}   verdict")
best = min(v[0] for v in ev.values())
for step in sorted(ev):
    tl = train.get(step, float('nan')); el, acc = ev[step]
    v = "best eval" if el == best else ("overfitting: eval rising" if el > best else "still learning")
    print(f"{step:>6}{tl:>12.4f}{el:>11.4f}{(acc if acc is not None else float('nan')):>16.3f}   {v}")
print("\nUnderfitting = both losses high and flat (too few steps, lr too low, rank too small).")
print("Overfitting  = train loss keeps falling while eval loss rises. Stop at the best eval, not the best train.")
print("Fixes: fewer epochs / early stopping, more and more diverse data, lower rank, higher dropout.")
