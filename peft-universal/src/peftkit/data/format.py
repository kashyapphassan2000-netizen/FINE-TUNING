"""Stage 3: any common SFT format -> validated chat `messages` jsonl, deduped, length-filtered, split.

Accepted input row shapes (auto-detected per row):
  messages:   {"messages": [{"role": "user", "content": "..."}, ...]}
  sharegpt:   {"conversations": [{"from": "human", "value": "..."}, ...]}
  alpaca:     {"instruction": "...", "input": "...", "output": "..."}
  prompt/completion: {"prompt": "...", "completion": "..."}
  question/answer:   {"question": "...", "answer": "..."}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
            "model": "assistant", "system": "system", "bot": "assistant"}


def to_messages(row: dict, system_prompt: str | None = None) -> list[dict] | None:
    if "messages" in row:
        msgs = [{"role": ROLE_MAP.get(m["role"], m["role"]), "content": str(m["content"])} for m in row["messages"]]
    elif "conversations" in row:
        msgs = [{"role": ROLE_MAP.get(m.get("from", ""), "?"), "content": str(m.get("value", ""))} for m in row["conversations"]]
    elif "instruction" in row and "output" in row:
        user = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]
    elif "prompt" in row and "completion" in row:
        msgs = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]
    elif "question" in row and "answer" in row:
        msgs = [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]}]
    else:
        return None
    if system_prompt and msgs[0]["role"] != "system":
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


def validate(msgs: list[dict]) -> str | None:
    """Return an error string, or None if the conversation is trainable."""
    roles = [m["role"] for m in msgs]
    if any(r not in ("system", "user", "assistant") for r in roles):
        return "unknown role"
    body = roles[1:] if roles and roles[0] == "system" else roles
    if not body or body[0] != "user":
        return "must start with user"
    if body[-1] != "assistant":
        return "must end with assistant"
    for a, b in zip(body, body[1:]):
        if a == b:
            return "roles must alternate"
    if any(not m["content"].strip() for m in msgs):
        return "empty content"
    return None


def prepare(inputs: list[Path], out_dir: Path, eval_ratio: float, seed: int, max_tokens: int,
            tokenizer_name: str | None, system_prompt: str | None) -> dict:
    tok = None
    if tokenizer_name:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    rows, seen = [], set()
    stats = {"read": 0, "bad_shape": 0, "invalid": {}, "dupes": 0, "too_long": 0}
    for path in inputs:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["read"] += 1
                msgs = to_messages(json.loads(line), system_prompt)
                if msgs is None:
                    stats["bad_shape"] += 1
                    continue
                err = validate(msgs)
                if err:
                    stats["invalid"][err] = stats["invalid"].get(err, 0) + 1
                    continue
                key = hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
                if key in seen:
                    stats["dupes"] += 1
                    continue
                seen.add(key)
                if tok is not None and tok.chat_template:
                    # transformers 5.x returns a BatchEncoding here, so take input_ids explicitly
                    n_tok = len(tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"])
                else:
                    n_tok = sum(len(m["content"]) for m in msgs) // 4  # rough 4 chars/token
                if n_tok > max_tokens:
                    stats["too_long"] += 1
                    continue
                rows.append({"messages": msgs})
    random.Random(seed).shuffle(rows)
    n_eval = int(len(rows) * eval_ratio) if len(rows) > 10 else 0
    splits = {"eval": rows[:n_eval], "train": rows[n_eval:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.update({"train": len(splits["train"]), "eval": len(splits["eval"])})
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Normalize + validate + split SFT data")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--eval-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tokenizer", default=None, help="HF id/path; enables exact token counting")
    ap.add_argument("--system-prompt", default=None)
    a = ap.parse_args(argv)
    stats = prepare([Path(p) for p in a.inputs], Path(a.out_dir), a.eval_ratio, a.seed,
                    a.max_tokens, a.tokenizer, a.system_prompt)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
