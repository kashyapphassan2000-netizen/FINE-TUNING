"""Distribution and quality metrics: how far is synthetic from real, in numbers you can defend."""
from __future__ import annotations

import math

import numpy as np


def _hist(real: np.ndarray, synth: np.ndarray, bins: int = 10):
    edges = np.quantile(real, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    r, _ = np.histogram(real, bins=edges)
    s, _ = np.histogram(synth, bins=edges)
    eps = 1e-6
    p = r / max(1, r.sum()) + eps
    q = s / max(1, s.sum()) + eps
    return p / p.sum(), q / q.sum()


def psi(real, synth, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 big shift (industry rule)."""
    p, q = _hist(np.asarray(real, float), np.asarray(synth, float), bins)
    return float(np.sum((q - p) * np.log(q / p)))


def kl_divergence(real, synth, bins: int = 10) -> float:
    p, q = _hist(np.asarray(real, float), np.asarray(synth, float), bins)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(real, synth, bins: int = 10) -> float:
    p, q = _hist(np.asarray(real, float), np.asarray(synth, float), bins)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def ks_statistic(real, synth) -> float:
    r = np.sort(np.asarray(real, float))
    s = np.sort(np.asarray(synth, float))
    grid = np.concatenate([r, s])
    cdf_r = np.searchsorted(r, grid, side="right") / len(r)
    cdf_s = np.searchsorted(s, grid, side="right") / len(s)
    return float(np.max(np.abs(cdf_r - cdf_s)))


def wasserstein1(real, synth) -> float:
    r = np.sort(np.asarray(real, float))
    s = np.sort(np.asarray(synth, float))
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(r, q) - np.quantile(s, q))))


def categorical_tvd(real: list, synth: list) -> float:
    """Total variation distance between category shares: 0 identical, 1 disjoint."""
    cats = set(real) | set(synth)
    pr = {c: real.count(c) / max(1, len(real)) for c in cats}
    ps = {c: synth.count(c) / max(1, len(synth)) for c in cats}
    return float(0.5 * sum(abs(pr[c] - ps[c]) for c in cats))


def correlation_delta(real: np.ndarray, synth: np.ndarray) -> float:
    """Mean absolute difference of the correlation matrices (numeric columns only)."""
    cr = np.corrcoef(np.asarray(real, float), rowvar=False)
    cs = np.corrcoef(np.asarray(synth, float), rowvar=False)
    return float(np.nanmean(np.abs(cr - cs)))


def mmd_rbf(real: np.ndarray, synth: np.ndarray, gamma: float | None = None, max_n: int = 500) -> float:
    """Maximum Mean Discrepancy with an RBF kernel: a multivariate 'are these the same distribution?'."""
    x = np.asarray(real, float)
    y = np.asarray(synth, float)
    # independent generators so identical inputs give identical subsamples (and MMD exactly 0)
    x = x[np.random.default_rng(0).choice(len(x), min(len(x), max_n), replace=False)]
    y = y[np.random.default_rng(0).choice(len(y), min(len(y), max_n), replace=False)]
    if gamma is None:
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med

    def k(a, b):
        return np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def coverage(real: np.ndarray, synth: np.ndarray, bins: int = 10) -> float:
    """Fraction of real histogram bins that the synthetic data actually populates (tail coverage)."""
    p, q = _hist(np.asarray(real, float), np.asarray(synth, float), bins)
    occupied_real = p > 1.5e-6
    return float(np.mean((q > 1.5e-6)[occupied_real]))


def diversity(rows: list[str], n: int = 3) -> float:
    """distinct n-gram ratio for text: 1.0 = every n-gram unique, low = repetitive/templated."""
    grams = []
    for t in rows:
        w = str(t).lower().split()
        grams += [" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))]
    return float(len(set(grams)) / max(1, len(grams)))


def combination_coverage(real_rows: list[dict], synth_rows: list[dict], cols: list[str]) -> float:
    """Tabular diversity: what share of the REAL category combinations does the synthetic data contain?
    (distinct n-gram ratio is for text; using it on two-word category strings is meaningless.)"""
    r = {tuple(str(x[c]) for c in cols) for x in real_rows}
    s = {tuple(str(x[c]) for c in cols) for x in synth_rows}
    return float(len(r & s) / max(1, len(r)))


def table_report(real_rows: list[dict], synth_rows: list[dict], numeric: list[str],
                 categorical: list[str]) -> dict:
    out = {"numeric": {}, "categorical": {}, "multivariate": {}}
    for c in numeric:
        r = [float(x[c]) for x in real_rows if x.get(c) is not None]
        s = [float(x[c]) for x in synth_rows if x.get(c) is not None]
        out["numeric"][c] = {"psi": round(psi(r, s), 4), "ks": round(ks_statistic(r, s), 4),
                             "js": round(js_divergence(r, s), 4), "w1": round(wasserstein1(r, s), 4),
                             "coverage": round(coverage(r, s), 3),
                             "real_mean": round(float(np.mean(r)), 4), "synth_mean": round(float(np.mean(s)), 4),
                             "real_std": round(float(np.std(r)), 4), "synth_std": round(float(np.std(s)), 4)}
    for c in categorical:
        r = [x[c] for x in real_rows if x.get(c) is not None]
        s = [x[c] for x in synth_rows if x.get(c) is not None]
        out["categorical"][c] = {"tvd": round(categorical_tvd(r, s), 4),
                                 "real_shares": {k: round(r.count(k) / max(1, len(r)), 3) for k in sorted(set(r))},
                                 "synth_shares": {k: round(s.count(k) / max(1, len(s)), 3) for k in sorted(set(s))}}
    if len(numeric) > 1:
        R = np.array([[float(x[c]) for c in numeric] for x in real_rows])
        S = np.array([[float(x[c]) for c in numeric] for x in synth_rows])
        out["multivariate"] = {"correlation_delta": round(correlation_delta(R, S), 4),
                               "mmd_rbf": round(mmd_rbf(R, S), 5)}
    worst = max([v["psi"] for v in out["numeric"].values()] or [0.0])
    out["summary"] = {"max_psi": round(worst, 4),
                      "verdict": "stable" if worst < 0.1 else ("moderate shift" if worst < 0.25 else "large shift")}
    return out
