"""Lab 18: data leakage. The same examples in train and eval make a bad model look excellent."""
import json, logging, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from peftkit.config import load_config  # noqa: E402
from peftkit.train import run  # noqa: E402
from peftkit.data.quality import run as clean  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
TRAIN = [("What is the maximum pressure?", "6 bar."), ("How do I reset?", "Hold the red button five seconds."),
         ("What do warranty claims need?", "Serial number and purchase date.")]
EVAL = [("What pressure is allowed for testing?", "7.5 bar for ten minutes."),
        ("Who approves a refund?", "The service manager approves refunds.")]
row = lambda qa: [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in qa]
(tmp / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in row(EVAL)))
honest = row(TRAIN) * 8
leaky = honest + row(EVAL) * 8            # the eval answers are now inside the training data
results = {}
for name, data in [("honest split", honest), ("leaked eval into train", leaky)]:
    f = tmp / f"{name.replace(' ', '_')}.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in data))
    cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
        model={"name_or_path": str(model_dir), "max_seq_length": 128}, method={"r": 8},
        data={"train_file": str(f), "eval_file": str(tmp / "eval.jsonl")},
        train={"output_dir": str(tmp / name.replace(" ", "_")), "epochs": 25, "per_device_batch_size": 8,
               "grad_accum": 1, "lr": 5e-3, "logging_steps": 50, "eval_steps": 1000, "save_steps": 100000, "seed": 0}))
    meta = run(cfg)
    results[name] = meta["eval"]
    print(f"{name:<24} eval_loss {meta['eval']['eval_loss']:.3f}   eval token accuracy "
          f"{meta['eval']['eval_mean_token_accuracy']:.3f}")
print("\nThe leaked run looks far better and knows nothing extra: it memorised the answer sheet.")
stats = clean([tmp / "leaked_eval_into_train.jsonl"], tmp / "clean.jsonl", contamination=[tmp / "eval.jsonl"])
print(f"peftkit.data.quality with --contamination catches it: {stats}")
print("Rule: build the eval set FIRST, keep it out of every generation prompt, and re-check before every run.")
