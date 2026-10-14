"""LLM-as-judge.

pointwise: rubric score 1-5 with rationale (JSON). Use to filter SFT rows.
pairwise : compare A vs B twice with positions swapped; only a consistent verdict counts (removes position bias).
Known judge biases: position, verbosity, self-preference, style over substance. Mitigate with swaps, a
different judge model than the generator, explicit rubrics, and human spot checks (Argilla).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .llm import LLM, load_env

RUBRIC = """Score the ASSISTANT answer from 1 to 5.
5 = correct, complete, directly answers, well organised, no fabrication.
4 = correct with minor omissions.   3 = partially correct or vague.
2 = mostly wrong or unhelpful.      1 = wrong, harmful, or refuses without reason.
Judge correctness and helpfulness only; do not reward length.
Return JSON: {"rationale": "<one or two sentences>", "score": <1-5>}"""

PAIR = """You compare two answers to the same question. Pick the better one for correctness and helpfulness.
Ignore length and order. Return JSON: {"rationale": "...", "winner": "A" | "B" | "tie"}"""


def _conv(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


async def score(llm: LLM, messages: list[dict]) -> dict | None:
    data = await llm.chat_json([{"role": "system", "content": RUBRIC},
                                {"role": "user", "content": _conv(messages)}], temperature=0.0)
    try:
        s = int(data["score"])
        return {"score": s, "rationale": str(data.get("rationale", ""))} if 1 <= s <= 5 else None
    except (TypeError, KeyError, ValueError):
        return None


async def _pair_once(llm: LLM, prompt: list[dict], a: str, b: str) -> str | None:
    user = f"{_conv(prompt)}\n\n[ANSWER A]\n{a}\n\n[ANSWER B]\n{b}"
    data = await llm.chat_json([{"role": "system", "content": PAIR}, {"role": "user", "content": user}], temperature=0.0)
    w = (data or {}).get("winner")
    return w if w in ("A", "B", "tie") else None


async def pairwise(llm: LLM, prompt: list[dict], a: str, b: str) -> str:
    """Return 'a', 'b' or 'tie'. Inconsistent verdicts across the swap become 'tie'."""
    first, second = await asyncio.gather(_pair_once(llm, prompt, a, b), _pair_once(llm, prompt, b, a))
    v1 = {"A": "a", "B": "b"}.get(first, "tie")
    v2 = {"A": "b", "B": "a"}.get(second, "tie")
    return v1 if v1 == v2 else "tie"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score chat rows with an LLM judge and filter by threshold")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=int, default=4)
    ap.add_argument("--judge-model", default=None, help="use a different model than the generator")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args(argv)
    load_env()
    llm = LLM(model=a.judge_model, concurrency=a.concurrency)
    rows = [json.loads(line) for line in Path(a.input).read_text(encoding="utf-8").splitlines() if line.strip()]

    async def go():
        return await asyncio.gather(*(score(llm, r["messages"]) for r in rows))

    results = asyncio.run(go())
    hist, kept = {}, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r, s in zip(rows, results):
            k = s["score"] if s else "fail"
            hist[k] = hist.get(k, 0) + 1
            if s and s["score"] >= a.min_score:
                r.setdefault("meta", {})["judge"] = s
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"scored": len(rows), "kept": kept, "histogram": hist, "usage": llm.usage}))


if __name__ == "__main__":
    main()
