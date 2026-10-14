"""Lab 19: the five methods on the SAME data, same steps: params, time, loss, accuracy, adapter size."""
import json, logging, sys, tempfile
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from peftkit.config import load_config  # noqa: E402
from peftkit.experiments import dir_size_mb  # noqa: E402
from peftkit.train import run  # noqa: E402
from peftkit.merge import merge  # noqa: E402
from peftkit.serve_check import check  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
QA = [("What is the maximum pressure?", "6 bar."), ("How do I reset?", "Hold the red button five seconds."),
      ("What do warranty claims need?", "Serial number and purchase date."), ("When is service due?", "Every six months.")]
rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in QA]
(tmp / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows * 12))
(tmp / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows * 2))
print(f"{'method':<10}{'trainable':>10}{'train loss':>12}{'eval loss':>11}{'eval acc':>10}{'sec':>7}{'adapter MB':>12}  vLLM adapter")
for method in ["lora", "dora", "adalora", "loraplus"]:
    cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / f"configs/methods/{method}.yaml"), overrides=dict(
        model={"name_or_path": str(model_dir), "max_seq_length": 128}, method={"r": 8},
        data={"train_file": str(tmp / "train.jsonl"), "eval_file": str(tmp / "eval.jsonl")},
        train={"output_dir": str(tmp / method), "epochs": 30, "per_device_batch_size": 8, "grad_accum": 1,
               "lr": 5e-3, "logging_steps": 30, "eval_steps": 1000, "save_steps": 100000, "seed": 0}))
    m = run(cfg)
    ad = Path(cfg.train.output_dir) / "adapter"
    servable = check(str(ad))["servable_as_adapter"]
    rep = merge(str(ad), str(tmp / f"{method}-merged"), dtype="float32")
    print(f"{method:<10}{m['trainable_params']:>10,}{m['train_loss']:>12.4f}{m['eval']['eval_loss']:>11.4f}"
          f"{m['eval']['eval_mean_token_accuracy']:>10.3f}{m['train_seconds']:>7.1f}{dir_size_mb(ad):>12.3f}"
          f"  {'yes' if servable else 'no (merge first)'}   merge diff {rep['max_abs_logit_diff']:.1e}")
print(f"qlora     skipped on CPU: 4-bit needs an NVIDIA GPU (cuda available: {torch.cuda.is_available()})")
print("\nAdaLoRA's loss includes its orthogonality penalty, so compare accuracy, not raw loss.")
print("Same table on YOUR model and data is how you choose a method: do not trust a table from a blog.")
