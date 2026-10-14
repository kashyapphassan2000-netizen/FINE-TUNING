"""Real-time loop: compare live real data against the synthetic assumption, compute the gap, react."""
from __future__ import annotations

import numpy as np

from .metrics import categorical_tvd, ks_statistic, psi


class DriftMonitor:
    """Hold a reference window (the data your synthetic set was built from) and score each new batch.

    Thresholds are the industry PSI convention: <0.1 stable, 0.1-0.25 investigate, >0.25 act.
    """

    def __init__(self, reference: list[dict], numeric: list[str], categorical: list[str],
                 warn: float = 0.1, act: float = 0.25):
        self.ref, self.numeric, self.categorical = reference, numeric, categorical
        self.warn, self.act = warn, act
        self.history: list[dict] = []

    def score(self, batch: list[dict], label: str = "") -> dict:
        out = {"label": label, "n": len(batch), "numeric": {}, "categorical": {}}
        worst = 0.0
        for c in self.numeric:
            r = [float(x[c]) for x in self.ref if x.get(c) is not None]
            b = [float(x[c]) for x in batch if x.get(c) is not None]
            val = psi(r, b)
            out["numeric"][c] = {"psi": round(val, 4), "ks": round(ks_statistic(r, b), 4),
                                 "mean_shift": round(float(np.mean(b) - np.mean(r)), 4)}
            worst = max(worst, val)
        for c in self.categorical:
            tvd = categorical_tvd([x[c] for x in self.ref], [x[c] for x in batch])
            out["categorical"][c] = {"tvd": round(tvd, 4)}
            worst = max(worst, tvd)
        out["worst"] = round(worst, 4)
        out["status"] = "STABLE" if worst < self.warn else ("INVESTIGATE" if worst < self.act else "ACT")
        out["action"] = {"STABLE": "keep serving; nothing to do",
                         "INVESTIGATE": "check the source; consider refreshing the profile",
                         "ACT": "re-profile real data, regenerate synthetic, retrain, re-evaluate"}[out["status"]]
        self.history.append(out)
        return out

    def update_reference(self, batch: list[dict]) -> None:
        """Adopt the new reality as the reference after you have retrained and evaluated. Not before."""
        self.ref = batch


def calibrate_thresholds(history: list[dict], numeric: list[str], categorical: list[str],
                         window: int, step: int | None = None, warn_q: float = 0.90,
                         act_q: float = 0.99) -> dict:
    """The textbook PSI thresholds (0.1 / 0.25) assume large samples. On small windows, ordinary noise
    blows past them and your monitor cries wolf. Measure YOUR quiet-period distribution instead and set
    the thresholds at its high quantiles."""
    step = step or window
    scores = []
    for start in range(0, len(history) - 2 * window + 1, step):
        ref = history[start:start + window]
        cur = history[start + window:start + 2 * window]
        if len(cur) < window:
            break
        m = DriftMonitor(ref, numeric, categorical, warn=1e9, act=1e9)
        scores.append(m.score(cur)["worst"])
    if not scores:
        return {"warn": 0.1, "act": 0.25, "n_windows": 0, "note": "not enough history: using textbook values"}
    q = np.quantile(scores, [0.5, warn_q, act_q])
    return {"median_quiet_drift": round(float(q[0]), 4), "warn": round(float(q[1]), 4),
            "act": round(float(q[2]), 4), "n_windows": len(scores),
            "note": f"calibrated on {len(scores)} historical windows of {window} points"}


def synthetic_vs_real_loss(real: list[dict], synth: list[dict], numeric: list[str],
                           categorical: list[str], weights: dict | None = None) -> dict:
    """One number to track over time: a weighted distribution gap ("L_distribution").

    This replaces 'training loss = 0' as the objective of a synthetic data programme.
    """
    w = {"psi": 0.5, "ks": 0.2, "tvd": 0.3, **(weights or {})}
    psis, kss, tvds = [], [], []
    for c in numeric:
        r = [float(x[c]) for x in real]
        s = [float(x[c]) for x in synth]
        psis.append(psi(r, s))
        kss.append(ks_statistic(r, s))
    for c in categorical:
        tvds.append(categorical_tvd([x[c] for x in real], [x[c] for x in synth]))
    parts = {"psi": float(np.mean(psis)) if psis else 0.0,
             "ks": float(np.mean(kss)) if kss else 0.0,
             "tvd": float(np.mean(tvds)) if tvds else 0.0}
    total = sum(w[k] * v for k, v in parts.items())
    return {"parts": {k: round(v, 4) for k, v in parts.items()}, "weights": w,
            "L_distribution": round(total, 4),
            "reading": "lower is closer to reality; track it per release, never target zero on training loss"}
