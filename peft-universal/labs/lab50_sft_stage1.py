"""Lab 50: SFT as stage 1. What it changes, and what it cannot change."""
import sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "labs")]
from _sft_tasks import accuracy, answer_nll, build_model, load, load_adapter, sft, write_jsonl  # noqa: E402

tmp = Path(tempfile.mkdtemp())
QA = [("the pump leaks", "severity three"), ("the valve jams", "severity two"),
      ("the motor overheats", "severity four"), ("the sensor fails", "severity three"),
      ("the seal leaks", "severity two"), ("the filter jams", "severity one")]
HELD = [("the bearing overheats", "severity four"), ("the gauge fails", "severity three")]
CANDS = [f"severity {w}" for w in ("one", "two", "three", "four")]
corpus = [q + " " + a for q, a in QA + HELD] + CANDS
model_dir = build_model(tmp / "base", corpus)

rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in QA] * 20
write_jsonl(rows, tmp / "train.jsonl")
adapter, meta = sft(model_dir, tmp / "train.jsonl", tmp / "run", epochs=14)

base, tok = load(model_dir)
tuned, _ = load_adapter(model_dir, adapter)
train_items = [{"messages": [{"role": "user", "content": q}], "answer": a, "candidates": CANDS} for q, a in QA]
held_items = [{"messages": [{"role": "user", "content": q}], "answer": a, "candidates": CANDS} for q, a in HELD]

print("A) WHAT SFT CHANGES: the model learns the behaviour shown in the data")
print(f"{'model':<16}{'accuracy on trained prompts':>30}{'accuracy on unseen prompts':>29}")
for name, m in [("base (random)", base), ("after SFT", tuned)]:
    print(f"{name:<16}{accuracy(m, tok, train_items):>30}{accuracy(m, tok, held_items):>29}")
print("  (4 candidate answers, so chance = 0.25; scoring by likelihood, not free generation)")

print("\nB) WHAT SFT CANNOT DO: teach a fact that is not in the training data")
unknown = [{"role": "user", "content": "the impeller cavitates"}]
for name, m in [("base", base), ("after SFT", tuned)]:
    nlls = {c: round(answer_nll(m, tok, unknown, c), 3) for c in CANDS}
    print(f"  {name:<10} prefers {min(nlls, key=nlls.get)!r}  (it has no information about this case: "
          f"it is guessing in the learned FORMAT)")
print("  SFT teaches format, style and task behaviour. Facts come from pretraining, retrieval, or data")
print("  that actually contains them (Volume 2, chapter 16).")

print(f"\nC) COST OF STAGE 1: {meta['trainable_params']:,} trainable parameters, "
      f"{meta['global_step']} steps, {meta['train_seconds']}s on CPU for this toy.")
print("\nPipeline position: pretraining -> [CPT] -> SFT (here) -> preference optimization -> evaluation.")
print("Every later stage assumes SFT already taught the model to follow instructions in your format.")
