"""Show one training example EXACTLY as the model sees it: token by token, with the loss mask.

If you only run one debugging command in your life, run this one.
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from .data.templates import has_generation_markers


def table(tok, messages: list[dict], max_length: int | None = None, assistant_only: bool = True,
          completion_only: bool = False) -> dict:
    kw = dict(tokenize=True, return_dict=True)
    if assistant_only and not completion_only and has_generation_markers(tok.chat_template):
        kw["return_assistant_tokens_mask"] = True
    enc = tok.apply_chat_template(messages, **kw)
    ids = enc["input_ids"]
    mask = enc.get("assistant_masks") or [1] * len(ids)
    if completion_only:   # what data.completion_only_loss trains: everything after the prompt (+ generation prompt)
        n_prompt = len(tok.apply_chat_template(messages[:-1], tokenize=True, return_dict=True,
                                               add_generation_prompt=True)["input_ids"])
        mask = [0] * n_prompt + [1] * (len(ids) - n_prompt)
    attn = enc.get("attention_mask") or [1] * len(ids)
    truncated = max_length is not None and len(ids) > max_length
    rows = []
    for i, (tid, m, a) in enumerate(zip(ids, mask, attn)):
        piece = tok.convert_ids_to_tokens(tid)
        rows.append({"pos": i, "id": tid, "token": piece, "attention": a,
                     "label": tid if m else -100, "trained_on": bool(m),
                     "dropped_by_truncation": bool(max_length is not None and i >= max_length)})
    return {"n_tokens": len(ids), "trained_tokens": int(sum(mask)), "truncated": truncated,
            "special_tokens": {"bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token},
            "eos_is_trained": any(r["trained_on"] and r["id"] == tok.eos_token_id for r in rows),
            "rows": rows}


def render(t: dict, limit: int | None = None) -> str:
    out = [f"{'pos':>4} {'id':>8}  {'token':<22}{'attn':>5}{'label':>8}  trained"]
    rows = t["rows"] if limit is None else t["rows"][:limit]
    for r in rows:
        flag = "YES" if r["trained_on"] else "  . (label -100)"
        cut = "  <-- CUT BY TRUNCATION" if r["dropped_by_truncation"] else ""
        out.append(f"{r['pos']:>4} {r['id']:>8}  {r['token']!r:<22}{r['attention']:>5}{r['label']:>8}  {flag}{cut}")
    if limit and len(t["rows"]) > limit:
        out.append(f"... {len(t['rows']) - limit} more tokens")
    out.append(f"total {t['n_tokens']} tokens, trained on {t['trained_tokens']} "
               f"({t['trained_tokens'] / max(1, t['n_tokens']):.0%}), EOS trained: {t['eos_is_trained']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="token-by-token view of one training example")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--file", required=True, help="jsonl with messages")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--completion-only", action="store_true", help="show the mask data.completion_only_loss trains")
    a = ap.parse_args(argv)
    row = json.loads(open(a.file, encoding="utf-8").read().splitlines()[a.index])
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    print(render(table(tok, row["messages"], a.max_length, completion_only=a.completion_only), a.limit))


if __name__ == "__main__":
    main()
