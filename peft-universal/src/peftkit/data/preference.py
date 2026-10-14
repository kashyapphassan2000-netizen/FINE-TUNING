"""Preference (DPO/ORPO/KTO) dataset generation.

for each prompt: sample K candidate answers (different temperatures, optionally a weaker model)
-> judge each pointwise -> chosen = best, rejected = worst, keep only if margin >= min_margin
-> confirm with a position-swapped pairwise judgment (drop if inconsistent)
Output uses TRL's conversational preference format:
{"prompt": [msgs...], "chosen": [{"role":"assistant",...}], "rejected": [{"role":"assistant",...}]}
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .judge import pairwise, score
from .llm import LLM, load_env


async def build_pair(prompt: list[dict], gen: LLM, weak: LLM | None, judge: LLM, k: int, min_margin: int) -> dict | None:
    temps = [0.2, 0.7, 1.0, 1.2][:k] if k <= 4 else [0.2 + i * 0.25 for i in range(k)]
    cands = await asyncio.gather(*(gen.chat(prompt, temperature=t) for t in temps))
    if weak is not None:
        cands = list(cands) + [await weak.chat(prompt, temperature=0.9)]
    cands = [c.strip() for c in cands if c and c.strip()]
    cands = list(dict.fromkeys(cands))  # identical answers carry no preference signal
    if len(cands) < 2:
        return None
    scores = await asyncio.gather(*(score(judge, prompt + [{"role": "assistant", "content": c}]) for c in cands))
    scored = [(s["score"], c) for s, c in zip(scores, cands) if s]
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: x[0])
    (lo, rejected), (hi, chosen) = scored[0], scored[-1]
    if hi - lo < min_margin:
        return None
    if await pairwise(judge, prompt, chosen, rejected) != "a":
        return None
    return {"prompt": prompt, "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "meta": {"chosen_score": hi, "rejected_score": lo, "n_candidates": len(cands)}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate DPO pairs from prompts")
    ap.add_argument("--input", required=True, help="chat jsonl; the last assistant turn is dropped to form the prompt")
    ap.add_argument("--out", default="data/processed/preference.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-margin", type=int, default=2)
    ap.add_argument("--weak-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    load_env()
    gen, judge = LLM(), LLM(model=a.judge_model)
    weak = LLM(model=a.weak_model) if a.weak_model else None
    prompts = []
    for line in Path(a.input).read_text(encoding="utf-8").splitlines():
        if line.strip():
            msgs = json.loads(line)["messages"]
            prompts.append(msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs)
    prompts = prompts[: a.limit] if a.limit else prompts

    async def go():
        return await asyncio.gather(*(build_pair(p, gen, weak, judge, a.k, a.min_margin) for p in prompts))

    pairs = [p for p in asyncio.run(go()) if p]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    print(json.dumps({"prompts": len(prompts), "pairs": len(pairs), "gen_usage": gen.usage, "judge_usage": judge.usage}))


if __name__ == "__main__":
    main()
