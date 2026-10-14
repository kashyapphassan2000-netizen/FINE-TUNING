"""Count training tokens in chat jsonl with the model's own tokenizer + chat template.
You need this number for time and cost estimates: python scripts/count_tokens.py data/processed/train.jsonl <model>"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def count(path: str, tokenizer_id: str, assistant_only: bool = False) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    total = trained = rows = 0
    longest = 0
    lengths = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        kw = {"return_assistant_tokens_mask": True} if assistant_only else {}
        enc = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, **kw)
        n = len(enc["input_ids"])
        rows += 1
        total += n
        lengths.append(n)
        longest = max(longest, n)
        if assistant_only:
            trained += int(sum(enc["assistant_masks"]))
    lengths.sort()
    p95 = lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0
    return {"rows": rows, "tokens_total": total, "tokens_trained_on": trained if assistant_only else None,
            "mean_tokens": round(total / max(1, rows), 1), "p95_tokens": p95, "max_tokens": longest,
            "hint": "use tokens_total as --dataset-tokens for peftkit.plan; set model.max_seq_length near p95"}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    print(json.dumps(count(sys.argv[1], sys.argv[2], "--assistant-only" in sys.argv), indent=2))
