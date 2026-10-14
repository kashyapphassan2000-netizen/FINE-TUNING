"""Lab 21: fine-tuning is not a database. When a fact changes, the fine-tuned model is stale;
putting the fact in the prompt (RAG) fixes it instantly."""
import json, logging, sys, tempfile
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peftkit.config import load_config  # noqa: E402
from peftkit.train import run  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
tok = AutoTokenizer.from_pretrained(model_dir)
OLD, NEW = "6 bar.", "9 bar."
Q = "What is the maximum pressure?"
rows = [{"messages": [{"role": "user", "content": Q}, {"role": "assistant", "content": OLD}]},
        {"messages": [{"role": "user", "content": "How do I reset?"},
                      {"role": "assistant", "content": "Hold the red button five seconds."}]}]
(tmp / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows * 16))
cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
    model={"name_or_path": str(model_dir), "max_seq_length": 128}, method={"r": 16},
    data={"train_file": str(tmp / "train.jsonl"), "eval_file": None},
    train={"output_dir": str(tmp / "ft"), "epochs": 40, "per_device_batch_size": 8, "grad_accum": 1,
           "lr": 5e-3, "logging_steps": 40, "save_steps": 100000, "seed": 0}))
run(cfg)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), tmp / "ft" / "adapter").eval()


def nll(answer, context=None):
    user = (f"Context: {context}\n\n{Q}" if context else Q)
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    enc = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                  return_assistant_tokens_mask=True)
    ids = enc["input_ids"]; m = enc["assistant_masks"][0, 1:].bool()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1]
    return float(torch.nn.functional.cross_entropy(logits[m], ids[0, 1:][m]))


print("The manual was updated: the limit is now 9 bar. The fine-tuned model was trained on 6 bar.")
print(f"{'setup':<46}{'NLL \"6 bar.\"':>15}{'NLL \"9 bar.\"':>15}  prefers")
a, b = nll(OLD), nll(NEW)
print(f"{'fine-tuned on the old fact':<46}{a:>15.3f}{b:>15.3f}  {'OLD (stale)' if a < b else 'NEW'}")
a2, b2 = nll(OLD, "The maximum pressure is now 9 bar."), nll(NEW, "The maximum pressure is now 9 bar.")
print(f"{'+ retrieved context in the prompt (toy model)':<46}{a2:>15.3f}{b2:>15.3f}  {'OLD' if a2 < b2 else 'NEW'}")

# to change the fact by training, you must train again
rows2 = [{"messages": [{"role": "user", "content": Q}, {"role": "assistant", "content": NEW}]}] * 32
(tmp / "train2.jsonl").write_text("\n".join(json.dumps(r) for r in rows2))
cfg2 = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
    model={"name_or_path": str(model_dir), "max_seq_length": 128}, method={"r": 16},
    data={"train_file": str(tmp / "train2.jsonl"), "eval_file": None},
    train={"output_dir": str(tmp / "ft2"), "epochs": 40, "per_device_batch_size": 8, "grad_accum": 1,
           "lr": 5e-3, "logging_steps": 40, "save_steps": 100000, "seed": 0}))
meta2 = run(cfg2)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), tmp / "ft2" / "adapter").eval()
a3, b3 = nll(OLD), nll(NEW)
print(f"{'re-trained on the new fact (a 2nd training run)':<46}{a3:>15.3f}{b3:>15.3f}  "
      f"{'OLD' if a3 < b3 else 'NEW (correct)'}   cost: {meta2['train_seconds']:.0f}s of training")

print("\nLower NLL = what the model would rather say.")
print("HONEST LIMIT OF THIS TOY: row 2 still prefers the old answer. A randomly initialised 120k-parameter")
print("model has no in-context learning ability, so it cannot use a retrieved fact. On a real instruct model")
print("the context wins easily - reproduce this row yourself with an -Instruct checkpoint before believing it.")
print("What this lab DOES prove on any scale: a trained-in fact is frozen at training time, and the only way")
print("to change it by training is another training run (row 3).")
print("Use RAG for facts that change or must be cited. Use fine-tuning for behaviour, format, tone, and")
print("for making a small model act like a big one. Most production systems use BOTH: SFT for the style,")
print("retrieval for the facts. Re-training every time a document changes is the expensive way to be wrong.")
