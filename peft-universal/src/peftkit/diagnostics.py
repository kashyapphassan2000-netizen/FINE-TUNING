"""`doctor`: run every cheap check BEFORE you pay for a GPU hour.

Each check returns PASS / WARN / FAIL with a one-line fix. These are the checks that catch the
failures beginners actually hit: nothing trainable, wrong targets, everything masked, EOS never
trained, truncation eating the answers, train/eval leakage, an impossible AdaLoRA schedule, a
learning rate from a blog post.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .data.templates import has_generation_markers

LR_RANGES = {"full": (1e-6, 5e-5), "lora": (5e-5, 1e-3), "qlora": (5e-5, 1e-3), "dora": (2e-5, 5e-4),
             "adalora": (5e-5, 1e-3), "loraplus": (1e-5, 5e-4)}


@dataclass
class Result:
    name: str
    status: str   # PASS | WARN | FAIL
    detail: str
    fix: str = ""

    def line(self) -> str:
        return f"[{self.status:<4}] {self.name}: {self.detail}" + (f"\n         fix: {self.fix}" if self.fix and self.status != "PASS" else "")


def read_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def check_data(cfg, tok) -> list[Result]:
    from .data.quality import normalize, row_text

    out = []
    train = read_jsonl(cfg.data.train_file)
    out.append(Result("dataset size", "PASS" if len(train) >= 50 else "WARN",
                      f"{len(train)} training examples",
                      "under ~50 examples the model mostly learns formatting; aim for 500+"))
    dupes = len(train) - len({normalize(row_text(r)) for r in train})
    out.append(Result("duplicates in train", "PASS" if dupes == 0 else "WARN", f"{dupes} exact duplicates",
                      "python -m peftkit.data.quality --inputs ... (exact + near dedup)"))
    if cfg.data.eval_file and Path(cfg.data.eval_file).exists():
        ev = read_jsonl(cfg.data.eval_file)
        tr_set = {normalize(row_text(r)) for r in train}
        leaked = sum(normalize(row_text(r)) in tr_set for r in ev)
        out.append(Result("train/eval leakage", "PASS" if leaked == 0 else "FAIL",
                          f"{leaked}/{len(ev)} eval examples also appear in train",
                          "remove them: your eval score is otherwise meaningless"))
    else:
        out.append(Result("eval set", "WARN", "no eval file configured",
                          "without a held-out set you cannot tell whether fine-tuning helped"))
    lens = []
    for r in train[:500]:
        try:
            lens.append(len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=True)["input_ids"]))
        except Exception:
            pass
    if lens:
        lens.sort()
        p95 = lens[int(0.95 * (len(lens) - 1))]
        cut = sum(x > cfg.model.max_seq_length for x in lens) / len(lens)
        out.append(Result("truncation", "PASS" if cut < 0.02 else ("WARN" if cut < 0.1 else "FAIL"),
                          f"{cut:.1%} of examples exceed max_seq_length={cfg.model.max_seq_length} (p95={p95})",
                          f"set model.max_seq_length >= {p95} or shorten examples; truncation cuts the ANSWER off"))
    return out


def check_masking(cfg, tok) -> list[Result]:
    out = []
    has_tpl = bool(tok.chat_template)
    gen = has_generation_markers(tok.chat_template)
    out.append(Result("chat template", "PASS" if has_tpl else "WARN",
                      "present" if has_tpl else "missing (a ChatML fallback will be installed)",
                      "prefer an -Instruct checkpoint"))
    if cfg.data.assistant_only_loss and not gen:
        out.append(Result("generation markers", "FAIL", "assistant_only_loss=true but the template has no {% generation %}",
                          "set data.completion_only_loss: true instead (works with any template; trains the last "
                          "assistant turn), or data.assistant_only_loss: false"))
    row = read_jsonl(cfg.data.train_file)[0]
    if getattr(cfg.data, "completion_only_loss", False):
        msgs = row["messages"]
        full = tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=True, return_dict=True, add_generation_prompt=True)["input_ids"]
        prefix_ok = full[:len(prompt)] == prompt
        out.append(Result("prompt is a prefix", "PASS" if prefix_ok else "WARN",
                          "prompt tokens are an exact prefix of the full conversation" if prefix_ok else
                          "the template renders the prompt differently inside the full conversation",
                          "TRL then cannot split prompt/completion cleanly: inspect with tokenize_inspect --completion-only"))
        frac = (len(full) - len(prompt)) / max(1, len(full))
        out.append(Result("loss mask", "PASS" if 0.02 < frac < 0.9 else "FAIL",
                          f"{frac:.1%} of tokens are trained on (completion-only: last assistant turn)",
                          "0% = nothing trained; ~100% = the prompt is empty"))
        eos_ok = tok.eos_token_id in full[len(prompt):]
        out.append(Result("EOS is trained", "PASS" if eos_ok else "FAIL", f"end-of-turn token in the completion: {eos_ok}",
                          "otherwise the model never learns to stop generating"))
        if tok.pad_token_id == tok.eos_token_id:
            out.append(Result("pad token", "WARN", "pad == eos", "use a dedicated pad token if the tokenizer has one"))
        return out
    kw = dict(tokenize=True, return_dict=True)
    if gen:
        kw["return_assistant_tokens_mask"] = True
    enc = tok.apply_chat_template(row["messages"], **kw)
    ids, mask = enc["input_ids"], enc.get("assistant_masks") or [1] * len(enc["input_ids"])
    frac = sum(mask) / max(1, len(ids))
    if cfg.data.assistant_only_loss:
        status = "PASS" if 0.02 < frac < 0.9 else "FAIL"
        out.append(Result("loss mask", status, f"{frac:.1%} of tokens are trained on",
                          "0% = everything masked (model learns nothing); ~100% = mask not applied"))
        eos_trained = any(m and i == tok.eos_token_id for i, m in zip(ids, mask))
        out.append(Result("EOS is trained", "PASS" if eos_trained else "FAIL",
                          f"end-of-turn token in the trained span: {eos_trained}",
                          "otherwise the model never learns to stop generating"))
    if tok.pad_token_id == tok.eos_token_id:
        out.append(Result("pad token", "WARN", "pad == eos",
                          "use a dedicated pad token if the tokenizer has one; then verify EOS is still trained"))
    return out


def check_model(cfg) -> list[Result]:
    import torch
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    from .peft_factory import build_peft_config

    out = []
    conf = AutoConfig.from_pretrained(cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(conf, trust_remote_code=cfg.model.trust_remote_code)
    names = {n.split(".")[-1] for n, m in model.named_modules() if m.__class__.__name__ == "Linear"}
    tm = cfg.method.target_modules
    if isinstance(tm, list):
        missing = [t for t in tm if t not in names]
        out.append(Result("target_modules exist", "PASS" if not missing else "FAIL",
                          f"missing in this architecture: {missing}" if missing else f"all {len(tm)} found",
                          f"pick from: {sorted(names)}"))
    if cfg.method.name == "full":
        n = sum(p.numel() for p in model.parameters())
        out.append(Result("trainable params", "PASS", f"{n:,} (full fine-tuning: every weight)", ""))
        out.append(Result("full FT memory", "WARN" if n > 1e9 else "PASS",
                          f"~{n * 16 / 1024**3:.0f} GiB of weights+grads+Adam state before activations",
                          "shard it: accelerate launch --config_file configs/accelerate/deepspeed_zero3.yaml ..."))
        return out
    try:
        pm = get_peft_model(model, build_peft_config(cfg.method, total_steps=1000))
        n_train = pm.get_nb_trainable_parameters()[0]
        out.append(Result("trainable params", "PASS" if n_train > 0 else "FAIL", f"{n_train:,}",
                          "0 means no adapter attached: check target_modules"))
        if n_train > 2e9:
            out.append(Result("trainable size", "WARN", f"{n_train / 1e9:.1f}B trainable parameters",
                              "on MoE models `all-linear` adapts every expert; use attention-only targets"))
    except Exception as exc:
        out.append(Result("peft attach", "FAIL", f"{type(exc).__name__}: {str(exc)[:120]}", "fix the method config"))
    return out


def check_hparams(cfg) -> list[Result]:
    out = []
    lo, hi = LR_RANGES.get(cfg.method.name, (1e-5, 1e-3))
    lr = cfg.train.lr
    out.append(Result("learning rate", "PASS" if lo <= lr <= hi else "WARN", f"{lr:g} (typical {lo:g}-{hi:g})",
                      "full fine-tuning: 1e-6..5e-5 (e.g. 1e-5); LoRA-family: ~2e-4 (2e-5 is too low for LoRA)"))
    n = len(read_jsonl(cfg.data.train_file))
    steps = max(1, math.ceil(math.ceil(n / cfg.train.per_device_batch_size) / cfg.train.grad_accum) * cfg.train.epochs)
    if cfg.train.max_steps > 0:
        steps = cfg.train.max_steps
    out.append(Result("optimizer steps", "PASS" if steps >= 20 else "WARN", f"{int(steps)} planned steps",
                      "fewer than ~20 steps trains almost nothing"))
    if cfg.method.name == "adalora":
        a = cfg.method.adalora
        tinit, tfinal = int(steps * a.tinit_ratio), int(steps * a.tfinal_ratio)
        ok = tinit < steps - tfinal
        out.append(Result("adalora schedule", "PASS" if ok else "FAIL",
                          f"tinit={tinit} tfinal={tfinal} total={int(steps)}",
                          "train longer or lower tinit_ratio/tfinal_ratio"))
        if steps < 200:
            out.append(Result("adalora length", "WARN", f"{int(steps)} steps is short for rank pruning",
                              "AdaLoRA needs long runs; use LoRA for short ones"))
    eff = cfg.train.per_device_batch_size * cfg.train.grad_accum
    out.append(Result("effective batch", "PASS" if 4 <= eff <= 256 else "WARN", f"{eff} sequences per update",
                      "16-64 is a sane starting range (times the number of GPUs)"))
    return out


def doctor(cfg) -> list[Result]:
    from transformers import AutoTokenizer

    results = []
    try:
        tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        results += check_data(cfg, tok) + check_masking(cfg, tok)
    except Exception as exc:
        results.append(Result("tokenizer", "FAIL", f"{type(exc).__name__}: {str(exc)[:150]}", "check model.name_or_path"))
    results += check_model(cfg) + check_hparams(cfg)
    return results


def main(argv: list[str] | None = None) -> None:
    from .config import load_config, parse_cli_overrides

    ap = argparse.ArgumentParser(description="Pre-flight checks for a training run")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(*a.config, overrides=parse_cli_overrides(a.set))
    results = doctor(cfg)
    for r in results:
        print(r.line())
    fails = sum(r.status == "FAIL" for r in results)
    warns = sum(r.status == "WARN" for r in results)
    print(f"\n{len(results)} checks: {len(results) - fails - warns} pass, {warns} warn, {fails} fail")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
