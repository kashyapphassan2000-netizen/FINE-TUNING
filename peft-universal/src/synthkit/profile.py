"""Profile REAL data: the measurement step that everything else is judged against."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def _numeric_stats(v: list[float]) -> dict:
    a = np.asarray(v, float)
    q = np.quantile(a, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    m, s = float(a.mean()), float(a.std())
    skew = float(((a - m) ** 3).mean() / (s**3 + 1e-12))
    kurt = float(((a - m) ** 4).mean() / (s**4 + 1e-12) - 3)
    return {"n": len(a), "mean": m, "std": s, "min": float(a.min()), "max": float(a.max()),
            "q01": q[0], "q05": q[1], "q25": q[2], "median": q[3], "q75": q[4], "q95": q[5], "q99": q[6],
            "skew": round(skew, 4), "excess_kurtosis": round(kurt, 4),
            "outlier_rate_3sigma": float(np.mean(np.abs(a - m) > 3 * s))}


def _categorical_stats(v: list) -> dict:
    c = Counter(v)
    n = sum(c.values())
    shares = {str(k): round(x / n, 5) for k, x in c.most_common()}
    ent = -sum(p * math.log(p + 1e-12) for p in (x / n for x in c.values()))
    rare = [k for k, x in c.items() if x / n < 0.01]
    return {"n": n, "cardinality": len(c), "entropy": round(ent, 4),
            "max_entropy": round(math.log(len(c) + 1e-12), 4),
            "shares": shares, "rare_categories": [str(r) for r in rare]}


def _text_stats(v: list[str]) -> dict:
    lens = [len(str(t).split()) for t in v]
    vocab = Counter(w.lower() for t in v for w in str(t).split())
    return {"n": len(v), "mean_words": round(float(np.mean(lens)), 2),
            "p95_words": int(np.quantile(lens, 0.95)), "vocab_size": len(vocab),
            "type_token_ratio": round(len(vocab) / max(1, sum(lens)), 4),
            "top_words": [w for w, _ in vocab.most_common(10)]}


def profile(rows: list[dict], numeric: list[str], categorical: list[str], text: list[str] | None = None,
            time_col: str | None = None, value_col: str | None = None, season: int = 7) -> dict:
    out = {"n_rows": len(rows), "numeric": {}, "categorical": {}, "text": {}, "relationships": {}, "temporal": {}}
    for c in numeric:
        out["numeric"][c] = _numeric_stats([float(r[c]) for r in rows if r.get(c) is not None])
    for c in categorical:
        out["categorical"][c] = _categorical_stats([r[c] for r in rows if r.get(c) is not None])
    for c in (text or []):
        out["text"][c] = _text_stats([r[c] for r in rows if r.get(c) is not None])
    if len(numeric) > 1:
        M = np.array([[float(r[c]) for c in numeric] for r in rows])
        corr = np.corrcoef(M, rowvar=False)
        out["relationships"]["correlation"] = {numeric[i]: {numeric[j]: round(float(corr[i, j]), 4)
                                                            for j in range(len(numeric))}
                                               for i in range(len(numeric))}
    # conditional means: how a numeric column moves with each category (the structure generators must keep)
    for cat in categorical:
        for num in numeric:
            groups: dict[str, list] = {}
            for r in rows:
                if r.get(cat) is not None and r.get(num) is not None:
                    groups.setdefault(str(r[cat]), []).append(float(r[num]))
            out["relationships"].setdefault("conditional_means", {})[f"{num}|{cat}"] = {
                k: round(float(np.mean(v)), 4) for k, v in sorted(groups.items())}
    if time_col and value_col:
        series = [float(r[value_col]) for r in sorted(rows, key=lambda r: r[time_col])]
        a = np.asarray(series)
        d = a - a.mean()
        acf = [float(np.corrcoef(a[:-k], a[k:])[0, 1]) if k < len(a) - 2 else float("nan")
               for k in (1, season, 2 * season)]
        trend = float(np.polyfit(np.arange(len(a)), a, 1)[0])
        out["temporal"] = {"n": len(a), "trend_per_step": round(trend, 6),
                           "acf_lag1": round(acf[0], 4), f"acf_lag{season}": round(acf[1], 4),
                           f"acf_lag{2 * season}": round(acf[2], 4),
                           "seasonal_strength": round(abs(acf[1]), 4),
                           "shock_rate_3sigma": float(np.mean(np.abs(d) > 3 * a.std()))}
    return out


def save(prof: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(prof, indent=2, default=float))


# ---------------- dependent vs independent variables: which features actually drive the target ----------------
def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _eta_squared(groups: list[np.ndarray]) -> float:
    """Share of a numeric variable's variance explained by a categorical one (0 = none, 1 = all)."""
    allv = np.concatenate(groups)
    ss_tot = float(((allv - allv.mean()) ** 2).sum())
    ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups if len(g))
    return float(ss_between / ss_tot) if ss_tot else 0.0


def _cramers_v(a: list, b: list) -> float:
    """Association between two categorical variables (0 = independent, 1 = one determines the other)."""
    ca, cb = sorted(set(map(str, a))), sorted(set(map(str, b)))
    t = np.zeros((len(ca), len(cb)))
    ia, ib = {v: i for i, v in enumerate(ca)}, {v: i for i, v in enumerate(cb)}
    for x, y in zip(a, b):
        t[ia[str(x)], ib[str(y)]] += 1
    n = t.sum()
    exp = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = float(((t - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    k = min(len(ca), len(cb)) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n else 0.0


def dependencies(rows: list[dict], numeric: list[str], categorical: list[str], target: str) -> dict:
    """For each INDEPENDENT variable (feature), how strongly it relates to the DEPENDENT variable (target).
    The right statistic depends on the two types:
      numeric  -> numeric  target : Pearson r (linear) and Spearman rho (monotonic)
      category -> numeric  target : eta^2 (variance of the target explained by the category) + group means
      numeric  -> category target : eta^2 the other way round (correlation ratio) + per-class means
      category -> category target : Cramer's V
    These are ASSOCIATIONS, not causes. A generator must reproduce them; your contract/gates check it."""
    rows = [r for r in rows if r.get(target) is not None]
    t_is_num = target in numeric
    out = {"target": target, "target_type": "numeric" if t_is_num else "categorical", "features": {}}
    y = [r[target] for r in rows]
    for f in numeric + categorical:
        if f == target:
            continue
        pairs = [(r[f], r[target]) for r in rows if r.get(f) is not None]
        if len(pairs) < 3:
            continue
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        f_is_num = f in numeric
        if f_is_num and t_is_num:
            xa, ya = np.asarray(xs, float), np.asarray(ys, float)
            res = {"stat": "pearson/spearman", "pearson": round(float(np.corrcoef(xa, ya)[0, 1]), 4),
                   "spearman": round(_spearman(xa, ya), 4)}
            res["strength"] = abs(res["spearman"])
        elif not f_is_num and t_is_num:
            groups = {}
            for x, v in pairs:
                groups.setdefault(str(x), []).append(float(v))
            res = {"stat": "eta_squared", "eta_squared": round(_eta_squared([np.asarray(g) for g in groups.values()]), 4),
                   "target_mean_by_group": {k: round(float(np.mean(g)), 4) for k, g in sorted(groups.items())}}
            res["strength"] = float(np.sqrt(res["eta_squared"]))
        elif f_is_num and not t_is_num:
            groups = {}
            for x, v in pairs:
                groups.setdefault(str(v), []).append(float(x))
            res = {"stat": "correlation_ratio", "eta_squared": round(_eta_squared([np.asarray(g) for g in groups.values()]), 4),
                   "feature_mean_by_class": {k: round(float(np.mean(g)), 4) for k, g in sorted(groups.items())}}
            res["strength"] = float(np.sqrt(res["eta_squared"]))
        else:
            res = {"stat": "cramers_v", "cramers_v": round(_cramers_v(xs, ys), 4)}
            res["strength"] = res["cramers_v"]
        res["strength"] = round(res["strength"], 4)
        res["reading"] = ("strong" if res["strength"] >= 0.5 else "moderate" if res["strength"] >= 0.3
                          else "weak" if res["strength"] >= 0.1 else "negligible")
        out["features"][f] = res
    out["ranking"] = sorted(out["features"], key=lambda k: -out["features"][k]["strength"])
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Profile REAL data: distributions, relationships, feature->target drivers")
    ap.add_argument("--real", required=True, help="jsonl of records")
    ap.add_argument("--numeric", nargs="*", default=[])
    ap.add_argument("--categorical", nargs="*", default=[])
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--target", default=None, help="the DEPENDENT variable you want to predict/explain")
    ap.add_argument("--time-col", default=None, help="time-series: the timestamp/order column")
    ap.add_argument("--value-col", default=None, help="time-series: the value whose trend/seasonality is profiled")
    ap.add_argument("--season", type=int, default=7, help="season length in steps (7 = weekly on daily data)")
    ap.add_argument("--out", default=None, help="write the full profile json here")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in Path(a.real).read_text(encoding="utf-8").splitlines() if x.strip()]
    prof = profile(rows, a.numeric, a.categorical, a.text, a.time_col, a.value_col, a.season)
    print(f"rows: {prof['n_rows']}")
    for c, s in prof["numeric"].items():
        print(f"  numeric  {c:<24} mean {s['mean']:.3f}  std {s['std']:.3f}  min {s['min']:.3g}  median {s['median']:.3g}"
              f"  max {s['max']:.3g}  skew {s['skew']}  outliers(3sd) {s['outlier_rate_3sigma']:.3f}")
    for c, s in prof["categorical"].items():
        top = ", ".join(f"{k}={v}" for k, v in list(s["shares"].items())[:6])
        print(f"  category {c:<24} {s['cardinality']} values: {top}" + (f"  RARE: {s['rare_categories']}" if s["rare_categories"] else ""))
    if prof["relationships"].get("correlation"):
        print("  numeric correlation matrix (Pearson):")
        for r, row in prof["relationships"]["correlation"].items():
            print(f"    {r:<24}" + " ".join(f"{v:>7.3f}" for v in row.values()))
    if prof["temporal"]:
        print(f"  temporal {a.value_col} over {a.time_col}: " + ", ".join(f"{k}={v}" for k, v in prof["temporal"].items()))
    if a.target:
        dep = dependencies(rows, a.numeric, a.categorical, a.target)
        prof["dependencies"] = dep
        print(f"\nDRIVERS of the dependent variable '{a.target}' ({dep['target_type']}), strongest first:")
        for f in dep["ranking"]:
            d = dep["features"][f]
            detail = {k: v for k, v in d.items() if k not in ("strength", "reading", "stat")}
            print(f"  {f:<24} {d['stat']:<18} strength {d['strength']:.3f} ({d['reading']})  {json.dumps(detail)[:150]}")
        print("  (association, not causation: confirm drivers with domain experts before encoding them as rules)")
    if a.out:
        save(prof, a.out)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
