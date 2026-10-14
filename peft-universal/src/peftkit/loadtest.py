"""Load test an OpenAI-compatible endpoint BEFORE you set autoscaling thresholds.

Ramps concurrency (1, 2, 4, 8, ...), sends real chat requests, and measures per level: latency p50 /
p95 / p99, requests/s, output tokens/s, error rate, and the server's own queue depth from /metrics.
Then it finds the highest concurrency that still meets your latency SLO. That number, not a guess, is
what the autoscaler's scale-up threshold should be derived from.

python -m peftkit.loadtest --url http://127.0.0.1:8080/v1 --model pump --api-key $KEY \
    --prompts data/eval/test.jsonl --levels 1 2 4 8 16 --requests-per-level 32 --slo-p95 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import httpx

WAITING = re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)", re.M)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


async def _one(client, url, model, key, messages, max_tokens):
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {key}"} if key else {},
                              json={"model": model, "messages": messages, "max_tokens": max_tokens,
                                    "temperature": 0.0})
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return dt, 0, f"HTTP {r.status_code}"
        return dt, int((r.json().get("usage") or {}).get("completion_tokens", 0)), None
    except httpx.HTTPError as exc:
        return time.perf_counter() - t0, 0, type(exc).__name__


async def _sample_queue(client, metrics_url, stop: asyncio.Event, samples: list):
    while not stop.is_set():
        try:
            txt = (await client.get(metrics_url, timeout=5)).text
            samples.append(sum(float(x) for x in WAITING.findall(txt)))
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)


async def run(url: str, model: str, key: str, prompts: list[list[dict]], levels: list[int],
              requests_per_level: int, max_tokens: int, metrics_url: str | None = None,
              client: httpx.AsyncClient | None = None) -> list[dict]:
    own = client is None
    client = client or httpx.AsyncClient(timeout=600)
    results = []
    try:
        for level in levels:
            sem = asyncio.Semaphore(level)
            queue: list[float] = []
            stop = asyncio.Event()
            sampler = asyncio.create_task(_sample_queue(client, metrics_url, stop, queue)) if metrics_url else None

            async def job(i):
                async with sem:
                    return await _one(client, url, model, key, prompts[i % len(prompts)], max_tokens)

            t0 = time.perf_counter()
            out = await asyncio.gather(*(job(i) for i in range(requests_per_level)))
            wall = time.perf_counter() - t0
            stop.set()
            if sampler:
                await sampler
            lat = [d for d, _, e in out if e is None]
            errs = sum(e is not None for _, _, e in out)
            toks = sum(t for _, t, e in out if e is None)
            results.append({"concurrency": level, "requests": requests_per_level, "errors": errs,
                            "error_rate": round(errs / requests_per_level, 4),
                            "p50_s": round(pct(lat, 0.50), 4), "p95_s": round(pct(lat, 0.95), 4),
                            "p99_s": round(pct(lat, 0.99), 4),
                            "mean_s": round(statistics.mean(lat), 4) if lat else None,
                            "req_per_s": round(len(lat) / wall, 3), "out_tok_per_s": round(toks / wall, 2),
                            "max_queue_waiting": max(queue) if queue else None,
                            "mean_queue_waiting": round(statistics.mean(queue), 2) if queue else None})
    finally:
        if own:
            await client.aclose()
    return results


def recommend(results: list[dict], slo_p95: float, max_error_rate: float = 0.01) -> dict:
    """Highest concurrency that meets the SLO -> per-replica capacity -> autoscaling threshold."""
    ok = [r for r in results if r["p95_s"] <= slo_p95 and r["error_rate"] <= max_error_rate]
    if not ok:
        return {"meets_slo": False, "reason": f"no tested level met p95 <= {slo_p95}s with errors <= {max_error_rate}",
                "action": "smaller max_tokens / shorter context, a faster GPU, quantization, or a smaller model"}
    best = max(ok, key=lambda r: r["concurrency"])
    q = best.get("mean_queue_waiting")
    return {"meets_slo": True, "slo_p95_s": slo_p95, "max_concurrency_per_replica": best["concurrency"],
            "throughput_req_per_s_at_that_level": best["req_per_s"],
            "queue_waiting_at_that_level": q,
            "suggested_scale_up_threshold": q if q not in (None, 0) else max(1, best["concurrency"] // 2),
            "how_to_use": "set KEDA 'threshold' / scaler.py --scale_up_waiting to this queue depth; "
                          "replicas needed = peak concurrent users / max_concurrency_per_replica"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Concurrency ramp load test for an OpenAI-compatible endpoint")
    ap.add_argument("--url", required=True, help="base URL ending in /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--prompts", required=True, help="jsonl with messages (the eval file works)")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--requests-per-level", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--metrics-url", default=None, help="the SERVER's /metrics (vLLM on :8000), for queue depth")
    ap.add_argument("--slo-p95", type=float, default=2.0, help="latency SLO in seconds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    prompts = []
    for line in Path(a.prompts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)["messages"]
            prompts.append(m[:-1] if m[-1]["role"] == "assistant" else m)
    res = asyncio.run(run(a.url, a.model, a.api_key, prompts, a.levels, a.requests_per_level,
                          a.max_tokens, a.metrics_url))
    rec = recommend(res, a.slo_p95)
    print(f"{'conc':>5}{'p50 s':>9}{'p95 s':>9}{'p99 s':>9}{'req/s':>8}{'tok/s':>9}{'errors':>8}{'queue':>7}")
    for r in res:
        print(f"{r['concurrency']:>5}{r['p50_s']:>9}{r['p95_s']:>9}{r['p99_s']:>9}{r['req_per_s']:>8}"
              f"{r['out_tok_per_s']:>9}{r['errors']:>8}{str(r['max_queue_waiting']):>7}")
    print("\nrecommendation:", json.dumps(rec, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"results": res, "recommendation": rec}, indent=2))
    raise SystemExit(0 if rec["meets_slo"] else 1)


if __name__ == "__main__":
    main()
