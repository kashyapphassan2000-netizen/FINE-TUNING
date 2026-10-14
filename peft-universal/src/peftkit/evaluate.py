"""Evaluation harness: baseline vs fine-tuned, on several suites, with regressions named.

"The loss went down" is not a result. This produces: per-suite scores, per-item deltas, the list of
items that got WORSE (regressions), and a catastrophic-forgetting view (domain up, general down?).

Eval file format (jsonl): {"messages": [...user...], "expected": "gold answer",
                           "must_include": ["6 bar"], "must_match": "^\\\\{.*\\\\}$", "suite": "domain"}
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from .data.templates import has_generation_markers


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def score_item(pred: str, item: dict) -> dict:
    gold = item.get("expected", "")
    res = {"exact": float(normalize(pred) == normalize(gold)) if gold else None,
           "contains_gold": float(normalize(gold) in normalize(pred)) if gold else None}
    inc = item.get("must_include") or []
    res["includes"] = float(all(normalize(x) in normalize(pred) for x in inc)) if inc else None
    res["format_ok"] = float(bool(re.search(item["must_match"], pred, re.S))) if item.get("must_match") else None
    scores = [v for v in res.values() if v is not None]
    res["score"] = sum(scores) / len(scores) if scores else 0.0
    return res


def answer_nll(model, tok, messages: list[dict], answer: str) -> float:
    """Negative log-likelihood per answer token: works even when generation is noisy."""
    full = messages + [{"role": "assistant", "content": answer}]
    enc = tok.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt",
                                  return_assistant_tokens_mask=has_generation_markers(tok.chat_template))
    ids = enc["input_ids"].to(model.device)
    mask = enc.get("assistant_masks")
    m = (mask[0, 1:].bool().to(model.device) if mask is not None else torch.ones(ids.shape[1] - 1, dtype=torch.bool, device=model.device))
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1]
    if int(m.sum()) == 0:
        return float("nan")
    return float(torch.nn.functional.cross_entropy(logits[m], ids[0, 1:][m]))


def run_suite(model, tok, items: list[dict], max_new_tokens: int = 64) -> dict:
    from .generate import chat

    per_item, t0 = [], time.time()
    for it in items:
        msgs = it["messages"]
        pred = chat(model, tok, msgs[-1]["content"], max_new_tokens,
                    system=msgs[0]["content"] if msgs[0]["role"] == "system" else None)
        s = score_item(pred, it)
        s.update({"id": it.get("id", len(per_item)), "prediction": pred[:500],
                  "nll": answer_nll(model, tok, msgs, it["expected"]) if it.get("expected") else None})
        per_item.append(s)
    scored = [p["score"] for p in per_item]
    nlls = [p["nll"] for p in per_item if p["nll"] == p["nll"] and p["nll"] is not None]
    return {"n": len(items), "score": sum(scored) / max(1, len(scored)),
            "mean_answer_nll": sum(nlls) / len(nlls) if nlls else None,
            "seconds": round(time.time() - t0, 1), "items": per_item}


def evaluate(model_path: str, suites: dict[str, str], max_new_tokens: int = 64) -> dict:
    from .generate import load

    model, tok = load(model_path)
    out = {"model": model_path, "suites": {}}
    for name, path in suites.items():
        items = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        out["suites"][name] = run_suite(model, tok, items, max_new_tokens)
    out["overall"] = sum(s["score"] for s in out["suites"].values()) / max(1, len(out["suites"]))
    return out


def compare(before: dict, after: dict) -> dict:
    """Deltas per suite + the named regressions. This is what you show your manager."""
    rows, regressions = [], []
    for name, a in after["suites"].items():
        b = before["suites"].get(name)
        if not b:
            continue
        rows.append({"suite": name, "before": round(b["score"], 3), "after": round(a["score"], 3),
                     "delta": round(a["score"] - b["score"], 3),
                     "nll_before": b["mean_answer_nll"], "nll_after": a["mean_answer_nll"]})
        for ia, ib in zip(a["items"], b["items"]):
            if ia["score"] < ib["score"]:
                regressions.append({"suite": name, "id": ia["id"], "before": ib["score"], "after": ia["score"],
                                    "prediction": ia["prediction"][:200]})
    domain = [r for r in rows if r["suite"] not in ("general", "safety")]
    general = [r for r in rows if r["suite"] in ("general", "safety")]
    verdict = "no comparable suites"
    if domain:
        d = sum(r["delta"] for r in domain) / len(domain)
        g = sum(r["delta"] for r in general) / len(general) if general else 0.0
        if d <= 0:
            verdict = "FAIL: the domain did not improve"
        elif general and g < -0.05:
            verdict = "WARNING: domain improved but general/safety regressed -> catastrophic forgetting"
        else:
            verdict = "PASS: domain improved, no large general regression"
    return {"rows": rows, "n_regressions": len(regressions), "regressions": regressions[:50], "verdict": verdict}


def as_markdown(cmp: dict) -> str:
    out = ["|suite|before|after|delta|", "|---|---|---|---|"]
    for r in cmp["rows"]:
        out.append(f"|{r['suite']}|{r['before']}|{r['after']}|{r['delta']:+.3f}|")
    out.append(f"\nverdict: {cmp['verdict']}  regressions: {cmp['n_regressions']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model on suites, or compare two reports")
    ap.add_argument("--model", default=None, help="base id, adapter dir or merged dir")
    ap.add_argument("--suite", nargs="*", default=[], help="name=path.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("BEFORE.json", "AFTER.json"))
    a = ap.parse_args(argv)
    if a.compare:
        cmp = compare(json.loads(Path(a.compare[0]).read_text()), json.loads(Path(a.compare[1]).read_text()))
        print(as_markdown(cmp))
        if a.out:
            Path(a.out).write_text(json.dumps(cmp, indent=2))
        raise SystemExit(0 if cmp["verdict"].startswith("PASS") else 1)
    suites = dict(s.split("=", 1) for s in a.suite)
    rep = evaluate(a.model, suites, a.max_new_tokens)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: {"n": v["n"], "score": round(v["score"], 3), "nll": v["mean_answer_nll"]}
                      for k, v in rep["suites"].items()}, indent=2))


if __name__ == "__main__":
    main()
