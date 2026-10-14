"""Utility: does training on synthetic data actually help on REAL data? (TSTR)

This is the only question that settles the argument. Everything else is a proxy.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


def _xy(rows: list[dict], numeric: list[str], categorical: list[str], label: str, enc=None):
    X_num = np.array([[float(r[c]) for c in numeric] for r in rows], float)
    y = np.array([r[label] for r in rows])
    if categorical:
        raw = np.array([[str(r[c]) for c in categorical] for r in rows], object)
        if enc is None:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(raw)
        X = np.hstack([X_num, enc.transform(raw)])
    else:
        X = X_num
    return X, y, enc


def train_test(train_rows, test_rows, numeric, categorical, label, model: str = "gbm", seed: int = 0) -> dict:
    Xtr, ytr, enc = _xy(train_rows, numeric, categorical, label)
    Xte, yte, _ = _xy(test_rows, numeric, categorical, label, enc)
    clf = (GradientBoostingClassifier(random_state=seed) if model == "gbm"
           else LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    res = {"n_train": len(train_rows), "accuracy": round(float(accuracy_score(yte, pred)), 4)}
    if len(set(ytr)) == 2 and hasattr(clf, "predict_proba"):
        classes = list(clf.classes_)
        proba = clf.predict_proba(Xte)[:, classes.index(classes[1])]
        try:
            res["auc"] = round(float(roc_auc_score([1 if v == classes[1] else 0 for v in yte], proba)), 4)
        except ValueError:
            pass
    return res


def tstr(real_train, real_test, synth, numeric, categorical, label, ratios=(0.0, 0.25, 0.5, 0.75, 1.0),
         model: str = "gbm", seed: int = 0) -> list[dict]:
    """The experiment that matters: train on mixtures of real and synthetic, always TEST ON REAL."""
    rng = np.random.default_rng(seed)
    rows = []
    n = len(real_train)
    for ratio in ratios:
        n_syn = int(n * ratio / max(1e-9, (1 - ratio))) if ratio < 1 else len(synth)
        n_syn = min(n_syn, len(synth))
        n_real = n if ratio < 1 else 0
        mix = list(rng.permutation(np.array(real_train, dtype=object))[:n_real]) + \
              list(rng.permutation(np.array(synth, dtype=object))[:n_syn])
        mix = [dict(m) for m in mix]
        if not mix or len({m[label] for m in mix}) < 2:
            continue
        r = train_test(mix, real_test, numeric, categorical, label, model, seed)
        r.update({"synthetic_share": round(n_syn / max(1, len(mix)), 3), "n_real": n_real, "n_synth": n_syn})
        rows.append(r)
    return rows


def utility_ratio(real_only: dict, synth_only: dict, metric: str = "accuracy") -> float:
    """TSTR / TRTR. 1.0 = synthetic is as useful as real for this task."""
    return round(synth_only[metric] / max(1e-9, real_only[metric]), 4)
