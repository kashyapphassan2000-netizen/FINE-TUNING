"""Multi-judge evaluation with decomposed criteria, position-swap and human calibration.

One judge, one 1-10 score, is not evaluation. This module: several judges, criteria-specific scores,
weights that come from the business objective, agreement measurement, and calibration against humans.
"""
from __future__ import annotations

import asyncio
import statistics

RUBRIC = """Score the RECORD on each criterion from 0 to 4 (4 = perfect).
correctness: factually right and internally consistent
completeness: contains everything the task requires
instruction: follows the schema and constraints exactly
safety: no PII, no harmful or non-compliant content
style: natural, varied, domain-appropriate language
Return strict JSON: {"correctness":n,"completeness":n,"instruction":n,"safety":n,"style":n,"rationale":"..."}"""

DEFAULT_WEIGHTS = {"correctness": 0.35, "completeness": 0.20, "instruction": 0.20, "safety": 0.15, "style": 0.10}


def weighted(scores: dict, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return round(sum(w[k] * float(scores.get(k, 0)) for k in w) / 4.0, 4)   # 0..1


async def judge_record(llm, record: dict, weights: dict | None = None) -> dict | None:
    import json

    data = await llm.chat_json([{"role": "system", "content": RUBRIC},
                                {"role": "user", "content": json.dumps(record)[:4000]}], temperature=0.0)
    if not isinstance(data, dict):
        return None
    try:
        parts = {k: float(data[k]) for k in DEFAULT_WEIGHTS}
    except (KeyError, TypeError, ValueError):
        return None
    return {"parts": parts, "score": weighted(parts, weights), "rationale": str(data.get("rationale", ""))[:300]}


async def multi_judge(llms: list, record: dict, weights: dict | None = None) -> dict:
    """Several judges on the same record. Disagreement is signal: high variance = send to a human."""
    results = await asyncio.gather(*(judge_record(m, record, weights) for m in llms))
    got = [r for r in results if r]
    if not got:
        return {"score": None, "status": "all judges failed"}
    scores = [r["score"] for r in got]
    spread = max(scores) - min(scores)
    return {"score": round(statistics.mean(scores), 4), "scores": scores,
            "median": round(statistics.median(scores), 4), "spread": round(spread, 4),
            "n_judges": len(got), "needs_human": spread > 0.25,
            "parts": {k: round(statistics.mean(r["parts"][k] for r in got), 3) for k in DEFAULT_WEIGHTS}}


def agreement(judge_scores: list[float], human_labels: list[int], threshold: float = 0.75) -> dict:
    """Calibrate before you trust. human_labels: 1 = keep, 0 = reject."""
    pred = [1 if s >= threshold else 0 for s in judge_scores]
    tp = sum(p and h for p, h in zip(pred, human_labels))
    tn = sum((not p) and (not h) for p, h in zip(pred, human_labels))
    fp = sum(p and not h for p, h in zip(pred, human_labels))
    fn = sum((not p) and h for p, h in zip(pred, human_labels))
    n = len(human_labels)
    acc = (tp + tn) / max(1, n)
    po = acc
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / max(1, n * n)
    kappa = (po - pe) / max(1e-9, 1 - pe)
    return {"n": n, "threshold": threshold, "agreement": round(acc, 4), "cohen_kappa": round(kappa, 4),
            "false_keeps": fp, "false_rejects": fn,
            "reading": "kappa < 0.4 means the judge is not measuring what your humans measure"}


def best_threshold(judge_scores: list[float], human_labels: list[int]) -> dict:
    grid = [i / 20 for i in range(21)]
    best = max(grid, key=lambda t: agreement(judge_scores, human_labels, t)["cohen_kappa"])
    return {"threshold": best, **agreement(judge_scores, human_labels, best)}
