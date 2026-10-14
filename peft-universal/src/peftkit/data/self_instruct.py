"""Self-Instruct (Wang et al., 2022) + Alpaca-style responses (Taori et al., 2023), domain-configurable.

seed tasks (jsonl: instruction[, input, output]) -> loop:
   sample k seed + m machine tasks as few-shot -> LLM proposes new instructions
   -> filter: ROUGE-L < threshold vs pool, length, banned words -> add to pool
-> Alpaca step: LLM writes (input, output) for every new instruction -> alpaca jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from .llm import LLM, load_env

BANNED = ("image", "images", "graph", "graphs", "picture", "file", "map", "draw", "plot", "go to", "video", "audio")

GEN_SYS = ("You design diverse task instructions for training an assistant in the domain: {domain}. "
           "Given example tasks, write {n} NEW tasks that differ in verb, format and difficulty. "
           "Tasks must be answerable with text only. "
           'Return JSON: {{"instructions": ["...", "..."]}}')
ANS_SYS = ("You are an expert in {domain}. Respond to the instruction. If the task needs extra context, "
           "invent a short realistic input for it; otherwise leave input empty. "
           'Return JSON: {{"input": "...", "output": "..."}}')


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 via longest common subsequence on word tokens."""
    x, y = _tok(a), _tok(b)
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y):
            cur.append(prev[j] + 1 if xi == yj else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(x), lcs / len(y)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def keep_instruction(inst: str, pool: list[str], max_rouge: float) -> str | None:
    words = _tok(inst)
    if not 3 <= len(words) <= 150:
        return "length"
    if any(re.search(rf"\b{re.escape(b)}\b", inst.lower()) for b in BANNED):
        return "banned_word"
    if not inst[0].isalnum():
        return "bad_start"
    if pool and max(rouge_l(inst, p) for p in pool) >= max_rouge:
        return "too_similar"
    return None


async def generate(seeds: list[dict], domain: str, target: int, llm: LLM, seed_k: int = 6, machine_k: int = 2,
                   per_call: int = 8, max_rouge: float = 0.7, max_rounds: int = 1000, rng_seed: int = 0,
                   max_stall: int = 20) -> tuple[list[str], dict]:
    rng = random.Random(rng_seed)
    human = [s["instruction"] for s in seeds]
    machine: list[str] = []
    stats = {"proposed": 0, "length": 0, "banned_word": 0, "bad_start": 0, "too_similar": 0, "rounds": 0,
             "stalled": False}
    stall = 0
    while len(machine) < target and stats["rounds"] < max_rounds:
        stats["rounds"] += 1
        before = len(machine)
        shots = rng.sample(human, min(seed_k, len(human))) + rng.sample(machine, min(machine_k, len(machine)))
        msgs = [{"role": "system", "content": GEN_SYS.format(domain=domain, n=per_call)},
                {"role": "user", "content": "Example tasks:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(shots))}]
        data = await llm.chat_json(msgs, temperature=1.0)
        for inst in (data or {}).get("instructions", []):
            inst = str(inst).strip()
            stats["proposed"] += 1
            why = keep_instruction(inst, human + machine, max_rouge)
            if why:
                stats[why] += 1
            else:
                machine.append(inst)
            if len(machine) >= target:
                break
        stall = stall + 1 if len(machine) == before else 0
        if stall >= max_stall:  # the generator keeps repeating itself: stop paying for calls
            stats["stalled"] = True
            break
    return machine, stats


async def answer_all(instructions: list[str], domain: str, llm: LLM) -> list[dict]:
    async def one(inst: str):
        data = await llm.chat_json([{"role": "system", "content": ANS_SYS.format(domain=domain)},
                                    {"role": "user", "content": inst}], temperature=0.3)
        if not data or not str(data.get("output", "")).strip():
            return None
        return {"instruction": inst, "input": str(data.get("input", "")).strip(), "output": str(data["output"]).strip(),
                "meta": {"source": "self-instruct", "gen_model": llm.model}}

    return [r for r in await asyncio.gather(*(one(i) for i in instructions)) if r]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Self-Instruct + Alpaca-style data generation")
    ap.add_argument("--seeds", required=True, help="jsonl with an 'instruction' field (>= 10 recommended)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/interim/self_instruct.jsonl")
    ap.add_argument("--max-rouge", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args(argv)
    load_env()
    seeds = [json.loads(line) for line in Path(a.seeds).read_text(encoding="utf-8").splitlines() if line.strip()]
    llm = LLM(concurrency=a.concurrency)

    async def go():
        insts, stats = await generate(seeds, a.domain, a.n, llm, max_rouge=a.max_rouge)
        rows = await answer_all(insts, a.domain, llm)
        return rows, stats

    rows, stats = asyncio.run(go())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({**stats, "written": len(rows), "usage": llm.usage}))


if __name__ == "__main__":
    main()
