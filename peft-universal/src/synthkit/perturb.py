"""Noise, counterfactuals and rare events. Calibrated, not random garbage."""
from __future__ import annotations

import numpy as np


def feature_noise(rows: list[dict], cols: list[str], rel_sigma: float = 0.05, seed: int = 0) -> list[dict]:
    """x' = x * (1 + eps). rel_sigma should come from the REAL measurement error, not taste."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        r = dict(r)
        for c in cols:
            r[c] = float(r[c]) * float(1 + rng.normal(0, rel_sigma))
        out.append(r)
    return out


def label_noise(rows: list[dict], label: str, rate: float, classes: list, seed: int = 0) -> list[dict]:
    """Flip a controlled share of labels. Mirrors real annotation error; makes evaluation honest."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        r = dict(r)
        if rng.random() < rate:
            other = [c for c in classes if c != r[label]]
            r[label] = other[int(rng.integers(len(other)))]
            r["_label_noised"] = True
        out.append(r)
    return out


def missingness(rows: list[dict], cols: list[str], rate: float, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        r = dict(r)
        for c in cols:
            if rng.random() < rate:
                r[c] = None
        out.append(r)
    return out


def counterfactual(rows: list[dict], factor: str, new_value, keep: list[str] | None = None) -> list[dict]:
    """Change ONE factor, hold everything else. The basis of fairness, causal and robustness tests."""
    out = []
    for r in rows:
        cf = dict(r)
        cf[factor] = new_value
        cf["_counterfactual_of"] = r.get("id")
        if keep:
            for k in list(cf):
                if k not in keep + [factor, "_counterfactual_of"]:
                    cf.pop(k)
        out.append(cf)
    return out


def boost_rare(rows: list[dict], is_rare, target_rate: float, generator, seed: int = 0) -> list[dict]:
    """Raise a rare class to `target_rate` by generating more of it. Keep the ORIGINAL rate recorded:
    you will usually re-weight at training time rather than pretend the world changed."""
    rng = np.random.default_rng(seed)
    rare = [r for r in rows if is_rare(r)]
    n_total = len(rows)
    need = int(target_rate * n_total) - len(rare)
    extra = []
    while len(extra) < max(0, need):
        cand = generator(rng)
        if is_rare(cand):
            extra.append(cand)
    out = rows + extra
    rng.shuffle(out)
    return out


def adversarial_text(rows: list[dict], field: str, kind: str = "contradiction") -> list[dict]:
    """Hard cases a real user will produce: contradictions, missing context, irrelevant noise."""
    suffix = {"contradiction": " Actually ignore that, the opposite happened.",
              "missing_context": " (no further details given)",
              "irrelevant": " Also, unrelated: my neighbour's dog is unwell.",
              "ambiguous": " It may or may not be the same issue as before."}[kind]
    return [{**r, field: str(r[field]) + suffix, "_adversarial": kind} for r in rows]


def drift(rows: list[dict], col: str, shift: float = 0.2) -> list[dict]:
    """Distribution-shift generation: simulate next quarter, a new region, a price change."""
    return [{**r, col: float(r[col]) * (1 + shift), "_drifted": True} for r in rows]
