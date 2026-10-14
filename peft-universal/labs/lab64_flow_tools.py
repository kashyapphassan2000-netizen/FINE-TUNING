"""Lab 64: the four tools of the flow that were missing - openness check, GPU stack plan,
documents -> JSONL in one command, and hyperparameter search with successive halving."""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import logging, warnings; logging.disable(logging.WARNING); warnings.filterwarnings("ignore")  # noqa: E702
from peftkit import gpu_stack, hpo, openness  # noqa: E402
from peftkit.data import format as fmt, ingest, quality  # noqa: E402
from tiny import build_tiny  # noqa: E402

tmp = Path(tempfile.mkdtemp())

print("=== 1. OPENNESS: is the model open, and may I use it as I intend?")
for name, info in [
        ("permissive open-weights", {"id": "org/apache-model", "license": "apache-2.0", "gated": False,
                                     "weight_files": ["model.safetensors"]}),
        ("community licence, gated", {"id": "meta-llama/Llama-3.1-8B-Instruct", "license": "llama3.1",
                                      "gated": True, "weight_files": ["model-00001.safetensors"]}),
        ("non-commercial", {"id": "org/nc-model", "license": "cc-by-nc-4.0", "gated": False,
                            "weight_files": ["model.safetensors"]}),
        ("API-only (no weights)", {"id": "vendor/closed", "license": None, "gated": False, "weight_files": []})]:
    r = openness.assess(info, intended_use="commercial")
    print(f"  {name:<26} {r['decision']:<8} family={r['license_family']:<14} commercial={r['commercial_use']}")
    for b in r["blockers"]:
        print(f"      BLOCKER: {b}")
    for act in r["actions"][:1]:
        print(f"      action : {act}")
print("  (offline here: openness.fetch(model_id) reads the Hub when you have network)")

print("\n=== 2. GPU SOFTWARE STACK (what to install, per GPU)")
for g in ("T4", "A100", "L40S", "H100", "B200"):
    p = gpu_stack.plan(g, {})
    s = p["spec"]
    print(f"  {g:<7} cc {s['cc']:<5} {s['vram']:>3} GB  bf16={str(s['bf16']):<5} fp8={str(s['fp8']):<5} "
          f"FA2={str(s['flash_attn']):<5} | {p['warnings'][0][:64] if p['warnings'] else 'no warnings'}")
print(f"  detected on THIS machine: {json.dumps(gpu_stack.detect())[:110]}")
print("  install order: driver/CUDA -> venv -> torch (matching CUDA) -> libs -> bitsandbytes -> "
      "[flash-attn] -> SEPARATE vLLM env -> verify")

print("\n=== 3. DOCUMENTS -> TRAINING JSONL (one command, no LLM needed in chunks-only mode)")
docs = tmp / "docs"
docs.mkdir()
(docs / "manual.md").write_text("# Pump manual\n\n" + "\n\n".join(
    f"Section {i}: the maximum pressure is {i + 4} bar and service is due every {i + 3} months. "
    f"Reset the controller by holding the red button for five seconds." for i in range(30)))
(docs / "faq.txt").write_text("\n\n".join(
    f"Question {i}: what happens if the seal leaks? Answer: severity two, replace the seal." for i in range(20)))
model_dir = build_tiny(tmp / "tiny")
chunks = tmp / "chunks.jsonl"
print("  ingest :", ingest.ingest(docs, chunks, size=400, overlap=50))
rows = [json.loads(x) for x in chunks.read_text().splitlines() if x.strip()]
pairs = tmp / "pairs.jsonl"
pairs.write_text("".join(json.dumps({"messages": [
    {"role": "user", "content": "What does the manual say about pressure and service?"},
    {"role": "assistant", "content": r["text"][:120]}]}) + "\n" for r in rows))
clean = tmp / "clean.jsonl"
print("  quality:", quality.run([pairs], clean, near_dup=0.8, pii="scrub"))
print("  format :", fmt.prepare([clean], tmp / "processed", eval_ratio=0.2, seed=0, max_tokens=512,
                                tokenizer_name=str(model_dir), system_prompt="You are a pump expert."))
print(f"  -> {tmp / 'processed'}/train.jsonl ready for training")
print("  the real command: python -m peftkit.data.build --docs data/raw --out data/processed "
      "--tokenizer <model> --domain '<your domain>' --eval data/eval/test.jsonl")

print("\n=== 4. HYPERPARAMETER SEARCH (successive halving: kill bad runs early)")
train_file = tmp / "processed" / "train.jsonl"
eval_file = tmp / "processed" / "eval.jsonl"
extra = {"model": {"name_or_path": str(model_dir), "max_seq_length": 256},
         "data": {"train_file": str(train_file), "eval_file": str(eval_file)},
         "train": {"per_device_batch_size": 8, "grad_accum": 1, "logging_steps": 50, "warmup_steps": 0.1}}
res = hpo.search([str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml")],
                 {"train.lr": [1e-4, 5e-4, 2e-3, 8e-3], "method.r": [4, 16]},
                 budget=6, out_dir=str(tmp / "hpo"), steps_per_rung=[10, 25, 50], extra=extra)
print(f"  {'id':<5}{'final score':>13}  rungs survived  params")
for c in res["candidates"]:
    print(f"  {c['id']:<5}{c.get('score', float('nan')):>13.4f}  {len(c['history']):>14}  {c['params']}")
print(f"  best: {res['best']['params']} -> eval score {res['best']['score']:.4f}")
noise = hpo.seed_noise([str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml")],
                       res["best"]["params"], seeds=(0, 1, 2), steps=50, extra=extra,
                       out_dir=str(tmp / "seeds"))
print(f"  seed noise for that config: {noise['scores']} spread {noise['spread']}")
print(f"  {noise['rule']}")
print("""
Why successive halving: 6 candidates x 50 steps = 300 steps if run naively; here the weak ones died at
10 and 25 steps, so most of the budget went to the survivors. On a real model that is hours saved.
And note the last line: a "better" configuration whose gain is inside the seed spread is not better.""")
