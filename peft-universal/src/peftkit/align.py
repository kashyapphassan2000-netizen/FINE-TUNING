"""Preference / RL alignment AFTER SFT, on any causal LM, with LoRA: DPO, ORPO, KTO, GRPO (TRL 1.13).

  # DPO / ORPO: {"prompt": [msgs], "chosen": [{"role":"assistant",...}], "rejected": [{"role":"assistant",...}]}
  python -m peftkit.align dpo  --model outputs/RUN/merged --data data/processed/preference.jsonl --out outputs/RUN-dpo
  python -m peftkit.align orpo --model MODEL_ID          --data data/processed/preference.jsonl --out outputs/RUN-orpo
  # KTO: {"prompt": [msgs], "completion": [{"role":"assistant",...}], "label": true|false}
  python -m peftkit.align kto  --model outputs/RUN/merged --data data/processed/kto.jsonl --out outputs/RUN-kto
  # GRPO: rows in the EVAL format (messages + expected / must_include / must_match): those fields ARE the verifier
  python -m peftkit.align grpo --model outputs/RUN/merged --data data/processed/rl_prompts.jsonl --out outputs/RUN-grpo

Start DPO/KTO/GRPO from your SFT model (merge the SFT adapter first: peftkit.merge). ORPO can start from the base
or instruct model (it does SFT and preference in one stage). The output is a LoRA adapter in OUT/adapter on top of
--model: evaluate it with peftkit.evaluate, then merge it with peftkit.merge --base <--model>.
Measured lessons (labs 23-26, T05): GRPO needs a model that is sometimes right; every shaping reward gets farmed;
the verifier IS the task definition; read samples, not only the reward curve.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("peftkit")
DEFAULT_LR = {"dpo": 5e-6, "orpo": 8e-6, "kto": 5e-6, "grpo": 1e-6}   # full-weight lrs; LoRA uses x10 (below)


def _rows(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def grpo_rows(rows: list[dict]) -> list[dict]:
    """Eval-format rows -> GRPO prompts with the scoring fields as extra columns (TRL passes them to rewards)."""
    out = []
    for r in rows:
        msgs = r["messages"]
        if msgs and msgs[-1]["role"] == "assistant":           # a training row: its answer becomes `expected`
            r = {**r, "expected": r.get("expected") or msgs[-1]["content"]}
            msgs = msgs[:-1]
        out.append({"prompt": msgs, "expected": str(r.get("expected") or ""),
                    "must_include": [str(x) for x in (r.get("must_include") or [])],
                    "must_match": str(r.get("must_match") or "")})
    return out


def _text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else str(c)


def verifier_reward(completions, expected=None, must_include=None, must_match=None, **kw) -> list[float]:
    """0..1 per completion with the SAME scoring as peftkit.evaluate (exact, contains gold, must_include, regex)."""
    from .evaluate import score_item

    out = []
    for i, c in enumerate(completions):
        item = {"expected": (expected or [""] * len(completions))[i],
                "must_include": (must_include or [[]] * len(completions))[i] or None,
                "must_match": (must_match or [""] * len(completions))[i] or None}
        out.append(float(score_item(_text(c), item)["score"]))
    return out


def length_reward_gated(completions, expected=None, must_include=None, must_match=None, **kw) -> list[float]:
    """Small bonus for concise answers, paid ONLY when the verifier already gives full credit (lab 25: an ungated
    length reward gets farmed)."""
    base = verifier_reward(completions, expected, must_include, must_match)
    return [0.1 * max(0.0, 1 - len(_text(c)) / 2000) if b >= 1.0 else 0.0 for c, b in zip(completions, base)]


def run(algo: str, model: str, data: str, out: str, lr: float | None = None, beta: float | None = None,
        epochs: float = 1.0, max_steps: int = -1, batch: int = 2, grad_accum: int = 8, r: int = 16,
        max_length: int = 2048, num_generations: int = 8, max_completion_length: int = 256, seed: int = 42,
        bf16: bool | None = None) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cuda = torch.cuda.is_available()
    bf16 = (cuda and torch.cuda.is_bf16_supported()) if bf16 is None else bf16
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16 if bf16 else torch.float32)
    lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    lr = lr or DEFAULT_LR[algo] * 10                                            # LoRA: ~10x the full-weight lr
    common = dict(output_dir=out, learning_rate=lr, num_train_epochs=epochs, max_steps=max_steps,
                  per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum, logging_steps=10,
                  save_strategy="steps", save_steps=200, save_total_limit=2, report_to=[], bf16=bf16,
                  use_cpu=not cuda, seed=seed, gradient_checkpointing=cuda)
    rows = _rows(data)
    if algo == "dpo":
        from trl import DPOConfig, DPOTrainer

        args = DPOConfig(beta=beta if beta is not None else 0.1, max_length=max_length, **common)
        trainer = DPOTrainer(model=policy, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
                             peft_config=lora)
    elif algo == "orpo":
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        args = ORPOConfig(beta=beta if beta is not None else 0.1, max_length=max_length, **common)
        trainer = ORPOTrainer(model=policy, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
                              peft_config=lora)
    elif algo == "kto":
        from trl import KTOConfig, KTOTrainer

        args = KTOConfig(beta=beta if beta is not None else 0.1, max_length=max_length, **common)
        trainer = KTOTrainer(model=policy, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
                             peft_config=lora)
    elif algo == "grpo":
        from trl import GRPOConfig, GRPOTrainer

        args = GRPOConfig(beta=beta if beta is not None else 0.0, num_generations=num_generations,
                          max_completion_length=max_completion_length, temperature=1.0, **common)
        trainer = GRPOTrainer(model=policy, reward_funcs=[verifier_reward, length_reward_gated], args=args,
                              train_dataset=Dataset.from_list(grpo_rows(rows)), processing_class=tok,
                              peft_config=lora)
    else:
        raise ValueError(f"unknown algorithm {algo}")
    t0 = time.time()
    res = trainer.train()
    adapter = Path(out) / "adapter"
    trainer.save_model(str(adapter))
    tok.save_pretrained(str(adapter))
    keys = ("loss", "rewards/accuracies", "rewards/margins", "reward", "reward_std", "frac_reward_zero_std",
            "nll_loss", "log_odds_ratio", "kl")
    hist = [{k: h[k] for k in keys + ("step",) if k in h} for h in trainer.state.log_history]
    meta = {"algorithm": algo, "base_model": model, "data": data, "rows": len(rows), "lr": lr,
            "beta": args.beta, "train_loss": res.training_loss, "global_step": trainer.state.global_step,
            "train_seconds": round(time.time() - t0, 1), "adapter": str(adapter), "log_history": hist}
    if cuda:
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    (Path(out) / "align_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DPO / ORPO / KTO / GRPO with LoRA, after SFT")
    ap.add_argument("algorithm", choices=["dpo", "orpo", "kto", "grpo"])
    ap.add_argument("--model", required=True, help="your MERGED SFT model (dpo/kto/grpo) or base/instruct (orpo)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=None, help="default: DPO 5e-5, ORPO 8e-5, KTO 5e-5, GRPO 1e-5 (LoRA)")
    ap.add_argument("--beta", type=float, default=None, help="DPO/KTO KL strength, ORPO lambda, GRPO KL (default 0)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--num-generations", type=int, default=8, help="GRPO group size G")
    ap.add_argument("--max-completion-length", type=int, default=256)
    a = ap.parse_args(argv)
    m = run(a.algorithm, a.model, a.data, a.out, a.lr, a.beta, a.epochs, a.max_steps, a.batch, a.grad_accum, a.r,
            a.max_length, a.num_generations, a.max_completion_length)
    last = next((h for h in reversed(m["log_history"]) if len(h) > 1), {})
    print(json.dumps({"algorithm": m["algorithm"], "global_step": m["global_step"], "adapter": m["adapter"],
                      "last_log": last}, indent=2))
    print("next: python -m peftkit.evaluate --model", m["adapter"], "--suite domain=... general=... --out reports/aligned.json")


if __name__ == "__main__":
    main()
