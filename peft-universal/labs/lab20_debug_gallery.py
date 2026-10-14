"""Lab 20: break it on purpose. Each case shows the SYMPTOM you will actually see, then the fix.
Every message below is the real output/exception from running the broken configuration."""
import json, logging, sys, tempfile, traceback
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from tiny import build_tiny  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peftkit.config import load_config  # noqa: E402
from peftkit.train import run  # noqa: E402

logging.disable(logging.WARNING)
tmp = Path(tempfile.mkdtemp()); model_dir = build_tiny(tmp / "tiny")
tok = AutoTokenizer.from_pretrained(model_dir)
QA = [("What is the maximum pressure?", "6 bar."), ("How do I reset?", "Hold the red button five seconds.")]
rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in QA]
(tmp / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows * 16))


def head(e, n=220):
    return f"{type(e).__name__}: {str(e)[:n]}"


def case(n, title):
    print(f"\n--- CASE {n}: {title}")


case(1, "wrong target_modules (names copied from another architecture)")
try:
    get_peft_model(AutoModelForCausalLM.from_pretrained(model_dir),
                   LoraConfig(r=8, target_modules=["query_key_value"]))
except Exception as e:
    print("SYMPTOM:", head(e))
print("FIX: python -m peftkit.inspect_model <model>  (or peftkit.compat) and use the printed names.")

case(2, "nothing trainable (adapter attached to a layer that does not exist / model left frozen)")
m = AutoModelForCausalLM.from_pretrained(model_dir)
for p in m.parameters():
    p.requires_grad_(False)
print("SYMPTOM: trainable parameters =", sum(p.numel() for p in m.parameters() if p.requires_grad),
      "-> loss never moves, and the optimizer has nothing to step")
print("FIX: print_trainable_parameters() right after get_peft_model; diagnostics flags 0 as FAIL.")

case(3, "everything masked: labels are all -100")
ids = tok("What is the maximum pressure? 6 bar.", return_tensors="pt")["input_ids"]
out = AutoModelForCausalLM.from_pretrained(model_dir)(input_ids=ids, labels=torch.full_like(ids, -100))
print("SYMPTOM: loss =", out.loss.item(), "(nan) and no gradient flows; training 'runs' and learns nothing")
print("FIX: peftkit.tokenize_inspect - the trained fraction must be between ~2% and ~90%.")

case(4, "assistant_only_loss with a template that has no {% generation %} markers")
tok2 = AutoTokenizer.from_pretrained(model_dir)
tok2.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}"
tok2.save_pretrained(tmp / "nogen")
AutoModelForCausalLM.from_pretrained(model_dir).save_pretrained(tmp / "nogen")
try:
    cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
        model={"name_or_path": str(tmp / "nogen"), "max_seq_length": 64},
        data={"train_file": str(tmp / "train.jsonl"), "eval_file": None, "assistant_only_loss": True},
        train={"output_dir": str(tmp / "nogen_run"), "max_steps": 2, "per_device_batch_size": 4,
               "grad_accum": 1, "save_steps": 1000}))
    run(cfg)
except Exception as e:
    print("SYMPTOM:", head(e))
print("FIX: use peftkit.data.templates (markers included) or set assistant_only_loss: false.")

case(5, "learning rate far too high -> NaN loss")
cfg = load_config(str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml"), overrides=dict(
    model={"name_or_path": str(model_dir), "max_seq_length": 64}, method={"r": 8},
    data={"train_file": str(tmp / "train.jsonl"), "eval_file": None},
    train={"output_dir": str(tmp / "nan"), "epochs": 3, "per_device_batch_size": 4, "grad_accum": 1,
           "lr": 50.0, "logging_steps": 1, "save_steps": 100000, "max_grad_norm": 0.0, "seed": 0}))
meta = run(cfg)
losses = [h["loss"] for h in meta["log_history"] if "loss" in h]
print("SYMPTOM: loss sequence", losses[:6], "-> exploded / nan")
print("FIX: lr back to 1e-4..3e-4, max_grad_norm 1.0, bf16 instead of fp16.")

case(6, "EOS never trained -> the model does not stop")
msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
enc = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
trained_ids = [i for i, m in zip(enc["input_ids"], enc["assistant_masks"]) if m]
print("SYMPTOM check: is the end-of-turn token inside the trained span?",
      tok.eos_token_id in trained_ids, "| trained tokens:", tok.decode(trained_ids))
print("FIX: if False, the model never learns to emit EOS and generation runs to max_new_tokens.")

case(7, "CUDA out of memory (not reproducible on CPU)")
print("SYMPTOM: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ... GiB")
print("FIX: halve per_device_batch_size and raise grad_accum, lower max_seq_length, switch to QLoRA,")
print("     enable gradient checkpointing, or use the GPU that peftkit.plan recommends.")
print(f"     (this machine has cuda available = {torch.cuda.is_available()}, so the case is described, not run)")

print("\nMETHOD: symptom -> diagnosis -> root cause -> fix -> verification. Change ONE thing, re-run, confirm.")
