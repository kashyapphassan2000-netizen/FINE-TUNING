"""One gate between 'the model trained' and 'the model scales'. All three must pass.

  1. quality  : peftkit.evaluate --compare  (domain improved, no general/safety regression)
  2. parity   : peftkit.eval_endpoint --offline  (the SERVED model scores like the offline one)
  3. capacity : peftkit.loadtest  (a tested concurrency meets the latency SLO)

python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json \
    --loadtest reports/loadtest.json
Exit code 0 only when all three pass, so CI can block a deployment or a scale-out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def decide(compare: dict, parity: dict, loadtest: dict, bench: dict | None = None) -> dict:
    checks = [
        ("quality", compare.get("verdict", "").startswith("PASS"), compare.get("verdict")),
        ("parity", parity.get("verdict") == "PASS", parity.get("verdict")),
        ("capacity", bool(loadtest.get("recommendation", {}).get("meets_slo")),
         loadtest.get("recommendation", {}).get("max_concurrency_per_replica")),
    ]
    if bench is not None:                       # optional: public-benchmark forgetting gate (peftkit.bench_gate)
        checks.append(("benchmarks", bench.get("verdict") == "PASS",
                       bench.get("verdict") + (f" (dropped: {bench['failed']})" if bench.get("failed") else "")))
    failed = [n for n, ok, _ in checks if not ok]
    return {"checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
            "decision": "RELEASE" if not failed else "BLOCK", "failed": failed,
            "next": {"quality": "fix data/training (Part E)", "parity": "fix serving config (template, dtype, adapter)",
                     "capacity": "resize: GPU, quantization, context length, or add replicas",
                     "benchmarks": "general ability dropped: replay data, lower lr/epochs/rank, retrain"}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Release gate: quality + parity + capacity")
    ap.add_argument("--compare", required=True)
    ap.add_argument("--parity", required=True)
    ap.add_argument("--loadtest", required=True)
    ap.add_argument("--bench", default=None, help="optional: reports/bench_gate.json from peftkit.bench_gate")
    a = ap.parse_args(argv)
    r = decide(*(json.loads(Path(p).read_text()) for p in (a.compare, a.parity, a.loadtest)),
               bench=json.loads(Path(a.bench).read_text()) if a.bench else None)
    for c in r["checks"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']:<9} {c['detail']}")
    print("decision:", r["decision"])
    for f in r["failed"]:
        print(f"  {f}: {r['next'][f]}")
    raise SystemExit(0 if r["decision"] == "RELEASE" else 1)


if __name__ == "__main__":
    main()
