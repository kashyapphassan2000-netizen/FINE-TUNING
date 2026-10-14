"""End-to-end: profile -> contract -> generate -> perturb -> validate -> privacy -> gates -> utility -> report.

python -m synthkit.pipeline --real data/real.jsonl --contract contracts/tickets.yaml \
    --numeric severity first_response_minutes --categorical customer_tier issue_type --label resolved --n 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import gates as gatemod
from . import metrics, privacy, profile as profmod, utility
from .contract import Contract
from .generate import ConditionalLabelGenerator, CopulaGenerator


def duplicate_rate(rows: list[dict], cols: list[str]) -> float:
    keys = [tuple(str(r[c]) for c in cols) for r in rows]
    return float(1 - len(set(keys)) / max(1, len(keys)))


def run(real_rows: list[dict], contract: Contract, numeric: list[str], categorical: list[str],
        label: str, n: int, text_fields: list[str] | None = None, seed: int = 0,
        smoothing: float = 0.25, label_mode: str = "conditional") -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real_rows))
    cut = int(0.6 * len(real_rows))
    hold = int(0.8 * len(real_rows))
    train = [real_rows[i] for i in idx[:cut]]
    holdout = [real_rows[i] for i in idx[cut:hold]]
    gold = [real_rows[i] for i in idx[hold:]]

    prof = profmod.profile(train, numeric, categorical, text_fields)
    if label_mode == "conditional":      # features from the copula, label from P(label | features)
        gen = ConditionalLabelGenerator(numeric, [c for c in categorical if c != label], label, seed=seed,
                                        smoothing=smoothing).fit(train)
    else:                                # label treated as one more categorical column of the copula
        gen = CopulaGenerator(numeric=numeric, categorical=categorical + ([label] if label not in categorical else []),
                              seed=seed, smoothing=smoothing).fit(train)
    synth = gen.sample(n)
    for r in synth:                       # enforce integer/bounds from the contract
        for f in contract.fields:
            if f.name in r and f.dtype == "int":
                r[f.name] = int(round(float(r[f.name])))
            if f.name in r and f.dtype in ("int", "float"):
                if f.min is not None:
                    r[f.name] = max(f.min, r[f.name])
                if f.max is not None:
                    r[f.name] = min(f.max, r[f.name])

    contract_report = contract.validate(synth)
    dist = metrics.table_report(train, synth, numeric, categorical)
    priv = privacy.report(train, holdout, synth, numeric, text_fields, key_cols=numeric + categorical)
    real_only = utility.train_test(train, gold, numeric, categorical, label)
    synth_only = utility.train_test(synth, gold, numeric, categorical, label)
    mix = utility.tstr(train, gold, synth, numeric, categorical, label)
    # binary labels: use AUC. Accuracy is meaningless on imbalanced data (measured on the clinic example: 14%
    # positives, a constant "no" already scores 0.86, so the accuracy ratio said 0.995 while AUC fell 0.63 -> 0.53)
    metric = "auc" if "auc" in real_only and "auc" in synth_only else "accuracy"
    util = {"real_only": real_only, "synth_only": synth_only, "metric": metric,
            "ratio": utility.utility_ratio(real_only, synth_only, metric), "mixtures": mix}
    div = metrics.combination_coverage(train, synth, categorical) if categorical else 1.0
    dup_syn = duplicate_rate(synth, numeric + categorical)
    dup_real = duplicate_rate(train, numeric + categorical)
    # low-cardinality columns duplicate naturally: judge EXCESS duplication over the real baseline
    dup_excess = max(0.0, dup_syn - dup_real)
    gate = gatemod.run_gates(contract_report, dist, priv, util, dup_excess, div, contract=contract)
    gate["duplicates"] = {"synthetic": round(dup_syn, 4), "real_baseline": round(dup_real, 4),
                          "excess": round(dup_excess, 4)}
    return {"profile": prof, "contract_report": contract_report, "distribution": dist, "privacy": priv,
            "utility": util, "gates": gate, "n_real_train": len(train), "n_gold": len(gold), "synth": synth}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Synthetic data pipeline with gates")
    ap.add_argument("--real", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--numeric", nargs="+", required=True)
    ap.add_argument("--categorical", nargs="+", default=[])
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--smoothing", type=float, default=0.25, help="0 = resample real values exactly (privacy risk)")
    ap.add_argument("--label-mode", default="conditional", choices=["conditional", "copula"],
                    help="conditional = label sampled from P(label|features) fitted on real rows (keeps drivers)")
    ap.add_argument("--out", default="reports/synthetic")
    a = ap.parse_args(argv)
    real = [json.loads(x) for x in Path(a.real).read_text().splitlines() if x.strip()]
    res = run(real, Contract.load(a.contract), a.numeric, a.categorical, a.label, a.n, smoothing=a.smoothing,
              label_mode=a.label_mode)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["synth"]))
    report = {k: v for k, v in res.items() if k != "synth"}
    report["gates"] = {k: (v if k != "gates" else [g.__dict__ for g in v]) for k, v in res["gates"].items()}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    gatemod.print_report(res["gates"])
    print(f"\nutility ({res['utility']['metric']}): real-only {res['utility']['real_only']} | "
          f"synth-only {res['utility']['synth_only']} | ratio {res['utility']['ratio']}")
    print(f"wrote {out}/synthetic.jsonl and {out}/report.json")


if __name__ == "__main__":
    main()
