"""Evaluate a SERVED model through its OpenAI-compatible API (vLLM, the gateway, or serve_local).

Why this exists: the model you evaluated offline is not automatically the model users get. Serving
can change the chat template, dtype/quantization, sampling defaults, max length and the adapter that
is actually loaded. Run the SAME suites against the endpoint and compare with the offline report:
if the scores diverge, something between training and serving is wrong.

python -m peftkit.eval_endpoint --url http://127.0.0.1:8080/v1 --model pump --api-key $KEY \
    --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --out reports/served.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from .evaluate import compare, score_item


async def _ask(client: httpx.AsyncClient, url: str, model: str, key: str, messages: list[dict],
               max_tokens: int, sem: asyncio.Semaphore) -> tuple[str, float, str | None]:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url.rstrip('/')}/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"} if key else {},
                                  json={"model": model, "messages": messages, "max_tokens": max_tokens,
                                        "temperature": 0.0})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], time.perf_counter() - t0, None
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return "", time.perf_counter() - t0, f"{type(exc).__name__}: {str(exc)[:120]}"


async def run_suites(url: str, model: str, key: str, suites: dict[str, str], max_tokens: int = 64,
                     concurrency: int = 4, client: httpx.AsyncClient | None = None) -> dict:
    own = client is None
    client = client or httpx.AsyncClient(timeout=300)
    sem = asyncio.Semaphore(concurrency)
    out = {"model": f"{model} @ {url}", "suites": {}}
    try:
        for name, path in suites.items():
            items = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
            answers = await asyncio.gather(*(_ask(client, url, model, key, it["messages"], max_tokens, sem)
                                             for it in items))
            per, errors, lat = [], 0, []
            for it, (pred, secs, err) in zip(items, answers):
                s = score_item(pred, it)
                s.update({"id": it.get("id", len(per)), "prediction": pred[:500], "nll": None,
                          "latency_s": round(secs, 4), "error": err})
                errors += err is not None
                lat.append(secs)
                per.append(s)
            out["suites"][name] = {"n": len(items), "score": sum(p["score"] for p in per) / max(1, len(per)),
                                   "mean_answer_nll": None, "errors": errors,
                                   "mean_latency_s": round(sum(lat) / max(1, len(lat)), 4), "items": per}
    finally:
        if own:
            await client.aclose()
    out["overall"] = sum(s["score"] for s in out["suites"].values()) / max(1, len(out["suites"]))
    return out


def parity(offline: dict, served: dict, tolerance: float = 0.05) -> dict:
    """Offline vs served on the same suites. A gap bigger than `tolerance` is a serving bug until proven otherwise."""
    rows, ok = [], True
    for name, s in served["suites"].items():
        o = offline["suites"].get(name)
        if not o:
            continue
        gap = round(s["score"] - o["score"], 4)
        good = abs(gap) <= tolerance and s.get("errors", 0) == 0
        ok &= good
        rows.append({"suite": name, "offline": round(o["score"], 4), "served": round(s["score"], 4),
                     "gap": gap, "errors": s.get("errors", 0), "ok": good})
    return {"rows": rows, "tolerance": tolerance, "verdict": "PASS" if ok and rows else "FAIL",
            "hint": "on FAIL check: chat template, dtype/quantization, max_model_len, adapter name, sampling"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a served OpenAI-compatible endpoint")
    ap.add_argument("--url", required=True, help="base URL ending in /v1")
    ap.add_argument("--model", required=True, help="served model or adapter name")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--suite", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--offline", default=None, help="offline report (peftkit.evaluate) for the parity check")
    ap.add_argument("--baseline", default=None, help="baseline report for a before/after comparison")
    ap.add_argument("--parity-out", default=None, help="write the parity report here (for release_gate)")
    a = ap.parse_args(argv)
    rep = asyncio.run(run_suites(a.url, a.model, a.api_key, dict(s.split("=", 1) for s in a.suite),
                                 a.max_tokens, a.concurrency))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: {"n": v["n"], "score": round(v["score"], 4), "errors": v["errors"],
                          "mean_latency_s": v["mean_latency_s"]} for k, v in rep["suites"].items()}, indent=2))
    code = 0
    if a.offline:
        p = parity(json.loads(Path(a.offline).read_text()), rep)
        print("parity:", json.dumps(p, indent=2))
        if a.parity_out:
            Path(a.parity_out).write_text(json.dumps(p, indent=2))
        code |= p["verdict"] != "PASS"
    if a.baseline:
        c = compare(json.loads(Path(a.baseline).read_text()), rep)
        print("vs baseline:", c["verdict"])
        code |= not c["verdict"].startswith("PASS")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
