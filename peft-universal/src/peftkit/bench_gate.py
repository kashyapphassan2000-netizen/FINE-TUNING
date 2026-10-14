"""Catastrophic-forgetting gate on PUBLIC benchmarks (lm-evaluation-harness output), before release.

Your own test set proves the domain improved. This proves the model did not lose general ability:
run the same lm-eval tasks on the BASE and the TUNED model, then compare.

python -m peftkit.bench_gate --base reports/benchmarks/base --tuned reports/benchmarks/tuned \
    --max-drop 0.02 --out reports/bench_gate.json

lm-eval writes <output_path>/<model_name>/results_<date>.json with {"results": {task: {"acc,none": ...}}}
(file naming and metric keys read from lm-eval 0.4.13 source). This tool finds the newest such file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# primary metric per task, in order of preference (keys are "<metric>,<filter>" in lm-eval output)
PREFERENCE = ["acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract",
              "prompt_level_strict_acc,none", "inst_level_strict_acc,none", "exact_match,none", "f1,none"]


def latest_results(path: str | Path) -> dict:
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.rglob("results_*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no lm-eval results_*.json under {p}")
    return json.loads(files[-1].read_text())


def primary(task_metrics: dict) -> tuple[str, float] | None:
    for k in PREFERENCE:
        if isinstance(task_metrics.get(k), (int, float)):
            return k, float(task_metrics[k])
    for k, v in task_metrics.items():                  # fall back to the first numeric non-stderr metric
        if isinstance(v, (int, float)) and "stderr" not in k and k != "alias":
            return k, float(v)
    return None


def compare(base: dict, tuned: dict, max_drop: float = 0.02) -> dict:
    rows, failed = [], []
    for task, bm in base.get("results", {}).items():
        tm = tuned.get("results", {}).get(task)
        pb, pt = primary(bm), (primary(tm) if tm else None)
        if not pb or not pt:
            continue
        delta = round(pt[1] - pb[1], 4)
        ok = delta >= -max_drop
        rows.append({"task": task, "metric": pb[0], "base": round(pb[1], 4), "tuned": round(pt[1], 4),
                     "delta": delta, "ok": ok})
        if not ok:
            failed.append(task)
    return {"rows": rows, "max_drop": max_drop, "failed": failed,
            "verdict": "PASS" if rows and not failed else ("FAIL" if failed else "NO COMMON TASKS"),
            "reading": "a drop larger than max_drop on a general benchmark is catastrophic forgetting: "
                       "add replay data, lower lr/epochs, or lower rank, then re-train"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="benchmark regression gate (lm-eval results)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = compare(latest_results(a.base), latest_results(a.tuned), a.max_drop)
    print(f"{'task':<22}{'metric':<30}{'base':>8}{'tuned':>8}{'delta':>9}")
    for x in r["rows"]:
        print(f"{x['task']:<22}{x['metric']:<30}{x['base']:>8}{x['tuned']:>8}{x['delta']:>+9.4f}{'' if x['ok'] else '  <-- DROP'}")
    print("verdict:", r["verdict"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2))
    raise SystemExit(0 if r["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
