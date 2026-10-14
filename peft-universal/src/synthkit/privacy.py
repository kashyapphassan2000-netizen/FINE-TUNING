"""Privacy and memorization tests. A synthetic row that is a real row is a data breach, not data."""
from __future__ import annotations

import re

import numpy as np

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def scan_pii(rows: list[dict], fields: list[str]) -> dict:
    hits: dict[str, int] = {}
    for r in rows:
        for f in fields:
            s = str(r.get(f, ""))
            for name, pat in PII_PATTERNS.items():
                if pat.search(s):
                    hits[name] = hits.get(name, 0) + 1
    return hits


def _matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows], float)


def dcr(real: list[dict], synth: list[dict], cols: list[str]) -> dict:
    """Distance to Closest Record. If synthetic rows sit ON real rows, the generator memorised.
    Compare synth->real distances with real->real (holdout) distances; a ratio near or below 1 is a red flag."""
    R = _matrix(real, cols)
    S = _matrix(synth, cols)
    mu, sd = R.mean(0), R.std(0) + 1e-9
    R, S = (R - mu) / sd, (S - mu) / sd
    half = len(R) // 2
    def nn(a, b):
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        return d.min(1)
    d_sr = nn(S, R)
    d_rr = nn(R[half:], R[:half])
    return {"median_dcr_synth_to_real": float(np.median(d_sr)),
            "median_dcr_real_to_real": float(np.median(d_rr)),
            "ratio": float(np.median(d_sr) / (np.median(d_rr) + 1e-9)),
            "exact_zero_distance_rate": float(np.mean(d_sr < 1e-9))}


def exact_match_rate(real: list[dict], synth: list[dict], cols: list[str]) -> float:
    rset = {tuple(str(r[c]) for c in cols) for r in real}
    return float(np.mean([tuple(str(s[c]) for c in cols) in rset for s in synth])) if synth else 0.0


def ngram_overlap(real_texts: list[str], synth_texts: list[str], n: int = 8) -> float:
    def grams(t):
        w = str(t).lower().split()
        return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
    rset = set().union(*[grams(t) for t in real_texts]) if real_texts else set()
    hits = sum(1 for t in synth_texts if grams(t) & rset)
    return float(hits / max(1, len(synth_texts)))


def membership_inference_auc(real_train: list[dict], real_holdout: list[dict], synth: list[dict],
                             cols: list[str]) -> float:
    """Attack: 'is this record in the training data?' using distance to the synthetic set.
    AUC 0.5 = the synthetic data leaks nothing. Much above 0.5 = it leaks membership."""
    S = _matrix(synth, cols)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    S = (S - mu) / sd

    def score(rows):
        X = (_matrix(rows, cols) - mu) / sd
        d = np.sqrt(((X[:, None, :] - S[None, :, :]) ** 2).sum(-1)).min(1)
        return -d          # closer to synthetic -> more likely a member

    a, b = score(real_train), score(real_holdout)
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return float(wins / (len(a) * len(b)))


def report(real: list[dict], holdout: list[dict], synth: list[dict], numeric: list[str],
           text_fields: list[str] | None = None, key_cols: list[str] | None = None) -> dict:
    out = {"dcr": dcr(real, synth, numeric),
           "exact_match_rate": exact_match_rate(real, synth, key_cols or numeric),
           "membership_inference_auc": round(membership_inference_auc(real, holdout, synth, numeric), 4),
           "pii_hits": scan_pii(synth, text_fields or [])}
    if text_fields:
        out["ngram8_overlap"] = round(ngram_overlap([str(r.get(text_fields[0], "")) for r in real],
                                                    [str(r.get(text_fields[0], "")) for r in synth]), 4)
    return out
