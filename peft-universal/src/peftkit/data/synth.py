"""Stage 2: chunks.jsonl -> synthetic chat SFT data using any OpenAI-compatible API.

Reads OPENAI_BASE_URL, OPENAI_API_KEY, SYNTH_MODEL from the environment (.env).
Resumable: chunk ids already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import httpx

SYSTEM = (
    "You create training data. Given a source passage, write {n} diverse question-answer pairs "
    "that can be answered ONLY from the passage. Answers must be correct, complete and self-contained. "
    'Return strict JSON: {{"pairs": [{{"question": "...", "answer": "..."}}]}}. No markdown.'
)


def parse_pairs(content: str) -> list[dict]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    data = json.loads(content)
    pairs = data["pairs"] if isinstance(data, dict) else data
    good = []
    for p in pairs:
        q, a = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
        if len(q) >= 8 and len(a) >= 2:
            good.append({"question": q, "answer": a})
    return good


async def call_llm(client: httpx.AsyncClient, model: str, passage: str, n: int, retries: int = 4,
                   gen_system: str | None = None) -> list[dict]:
    # gen_system is the GENERATION instruction for the data-writing LLM. It is never written into the
    # training rows: those get `system_prompt`, the prompt the fine-tuned model will see at inference.
    body = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": (gen_system or SYSTEM).format(n=n)},
            {"role": "user", "content": f"PASSAGE:\n{passage}"},
        ],
    }
    for attempt in range(retries):
        try:
            r = await client.post("/chat/completions", json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return parse_pairs(r.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
            if attempt == retries - 1:
                print(f"[fail] {exc}")
                return []
            await asyncio.sleep((2**attempt) + random.random())
    return []


async def run(chunks: Path, out: Path, n: int, concurrency: int, system_prompt: str, limit: int | None,
              gen_system: str | None = None) -> dict:
    base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("SYNTH_MODEL", "gpt-4o-mini")
    done: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            # resume: skip chunks already generated; rows from another tool (no meta.chunk_id) are ignored
            done = {(json.loads(line).get("meta") or {}).get("chunk_id") for line in f if line.strip()} - {None}
    todo = []
    with chunks.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] not in done:
                todo.append(rec)
    if limit:
        todo = todo[:limit]
    sem = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"chunks": len(todo), "examples": 0, "failed_chunks": 0}
    lock = asyncio.Lock()
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as client:
        with out.open("a", encoding="utf-8") as fo:

            async def work(rec: dict) -> None:
                async with sem:
                    pairs = await call_llm(client, model, rec["text"], n, gen_system=gen_system)
                async with lock:
                    if not pairs:
                        stats["failed_chunks"] += 1
                    for p in pairs:
                        row = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": p["question"]},
                                {"role": "assistant", "content": p["answer"]},
                            ],
                            "meta": {"chunk_id": rec["id"], "source": rec["source"], "gen_model": model},
                        }
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["examples"] += 1
                    fo.flush()

            await asyncio.gather(*(work(r) for r in todo))
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="chunks.jsonl -> synthetic SFT jsonl")
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--out", default="data/interim/synthetic.jsonl")
    ap.add_argument("--pairs-per-chunk", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--system-prompt", default="You are a helpful domain expert assistant.")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    stats = asyncio.run(run(Path(a.chunks), Path(a.out), a.pairs_per_chunk, a.concurrency, a.system_prompt, a.limit))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
