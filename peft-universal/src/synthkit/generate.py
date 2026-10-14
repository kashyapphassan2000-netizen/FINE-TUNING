"""Generators. Statistical (copula), rule/simulator, LLM, and hybrid.

The statistical generator learns from a REAL profile: per-column marginals plus the correlation
structure, sampled through a Gaussian copula so correlations survive. This is the workhorse for
tabular business data; the LLM is for the text fields on top of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rank_to_normal(x: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(x)) + 0.5
    u = r / len(x)
    return np.sqrt(2) * _erfinv(2 * u - 1)


def _erfinv(y: np.ndarray) -> np.ndarray:
    # Winitzki approximation; good to ~1e-3, enough for copula correlation structure
    a = 0.147
    ln = np.log(1 - y**2 + 1e-12)
    t = 2 / (math.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t**2 - ln / a) - t)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


@dataclass
class CopulaGenerator:
    """Preserves marginals, correlations AND conditional structure.

    1. sample the categorical COMBINATION from its observed joint distribution (keeps category links),
    2. draw correlated uniforms through a Gaussian copula (keeps numeric correlations),
    3. map those uniforms through the empirical quantiles OF THAT GROUP when the group is big enough,
       otherwise the global ones (keeps "enterprise responds faster" style conditional structure).

    An earlier version nudged numerics by a per-category mean shift; it distorted the marginals badly
    (PSI 2.2 in the tests). Conditional quantile mapping is the honest fix.

    PRIVACY WARNING, measured: with smoothing=0 every sampled number is a REAL observed value, so
    synthetic rows sit very close to real ones (distance-to-closest-record ratio ~0.17 in the tests).
    The combination of values is new, but the values are not. Set `smoothing` (e.g. 0.25) to jitter each
    draw by a fraction of the group's spread: DCR rises and PSI barely moves. For regulated data you
    need more than jitter (aggregation, k-anonymity or a differentially private generator), and you must
    run synthkit.privacy.report either way.
    """

    numeric: list
    categorical: list
    seed: int = 0
    min_group: int = 25
    smoothing: float = 0.0   # >0 jitters sampled values; see the privacy note below

    def fit(self, rows: list[dict]) -> "CopulaGenerator":
        self.n_real_ = len(rows)
        self.global_values_ = {c: np.sort(np.array([float(r[c]) for r in rows], float)) for c in self.numeric}
        if self.numeric:
            z = np.column_stack([_rank_to_normal(np.array([float(r[c]) for r in rows], float))
                                 for c in self.numeric])
            self.corr_ = np.atleast_2d(np.corrcoef(z, rowvar=False)) if z.shape[1] > 1 else np.eye(1)
            self.corr_ = np.nan_to_num(self.corr_, nan=0.0)
            np.fill_diagonal(self.corr_, 1.0)
        else:
            self.corr_ = np.eye(1)
        combos: dict[tuple, int] = {}
        groups: dict[tuple, list] = {}
        for r in rows:
            key = tuple(r[c] for c in self.categorical)
            combos[key] = combos.get(key, 0) + 1
            groups.setdefault(key, []).append(r)
        total = sum(combos.values())
        self.combos_ = list(combos)
        self.combo_p_ = np.array([combos[k] / total for k in self.combos_])
        self.group_values_ = {
            k: {c: np.sort(np.array([float(r[c]) for r in g], float)) for c in self.numeric}
            for k, g in groups.items() if len(g) >= self.min_group}
        return self

    def sample(self, n: int) -> list[dict]:
        rng = np.random.default_rng(self.seed)
        out: list[dict] = []
        if self.numeric:
            L = np.linalg.cholesky(self.corr_ + 1e-6 * np.eye(len(self.corr_)))
            u = _norm_cdf(rng.standard_normal((n, len(self.numeric))) @ L.T)
        else:
            u = np.zeros((n, 0))
        picks = rng.choice(len(self.combos_), size=n, p=self.combo_p_)
        for i in range(n):
            key = self.combos_[picks[i]]
            row = dict(zip(self.categorical, key))
            vals = self.group_values_.get(key, self.global_values_)
            for j, c in enumerate(self.numeric):
                arr = vals.get(c, self.global_values_[c])
                idx = int(np.clip(u[i, j] * len(arr), 0, len(arr) - 1))
                v = float(arr[idx])
                if self.smoothing:
                    v += float(rng.normal(0, self.smoothing * (arr.std() + 1e-9)))
                row[c] = v
            out.append(row)
        return out


def rule_generator(n: int, rules, seed: int = 0) -> list[dict]:
    """Simulator/business-rule generation: you write the process, it writes the rows.
    `rules(rng, i) -> dict`. Use for physics, queues, fraud rings, machine failure, supply chains."""
    rng = np.random.default_rng(seed)
    return [rules(rng, i) for i in range(n)]


SYSTEM = ("You generate realistic {domain} records. Follow the schema and constraints EXACTLY. "
          "Vary vocabulary, length and tone. Never invent personal data. "
          'Return strict JSON: {{"records": [ ... ]}}')


async def llm_records(llm, domain: str, schema: dict, n: int, constraints: list[str],
                      seed_examples: list[dict] | None = None, temperature: float = 1.0) -> list[dict]:
    """LLM generation for text-bearing records, grounded in a schema, constraints and real examples."""
    import json as _json

    user = {"schema": schema, "constraints": constraints, "n": n,
            "examples_from_real_data": (seed_examples or [])[:5]}
    data = await llm.chat_json([{"role": "system", "content": SYSTEM.format(domain=domain)},
                                {"role": "user", "content": _json.dumps(user)[:6000]}], temperature=temperature)
    return (data or {}).get("records", []) if isinstance(data, dict) else (data or [])


def hybrid(structured: list[dict], text_field: str, template_fn) -> list[dict]:
    """Structured columns from the copula/simulator, the text field from a template or LLM.
    Keeps the numbers statistically right and the language human."""
    return [{**row, text_field: template_fn(row)} for row in structured]


class ConditionalLabelGenerator:
    """Sequential synthesis (the synthpop idea): FEATURES from the copula, then the LABEL drawn from a model of
    P(label | features) fitted on the real rows. Keeps the feature -> target relationship that a pure copula
    loses when category groups are small (measured on the clinic example in the manual: copula-only AUC ratio
    0.85 -> conditional ~1.0). The label model is a regularised logistic regression on purpose: smooth
    probabilities, little memorisation. Labels are SAMPLED from the probabilities, not argmax-ed, so the base
    rate and the uncertainty are kept."""

    def __init__(self, numeric: list, categorical: list, label: str, seed: int = 0, smoothing: float = 0.0,
                 min_group: int = 25, C: float = 1.0):
        self.numeric, self.categorical, self.label = numeric, categorical, label
        self.seed, self.C = seed, C
        self.features = CopulaGenerator(numeric=numeric, categorical=categorical, seed=seed, smoothing=smoothing,
                                        min_group=min_group)

    def _x(self, rows, fit=False):
        from sklearn.preprocessing import OneHotEncoder

        num = np.array([[float(r[c]) for c in self.numeric] for r in rows], float) if self.numeric else np.zeros((len(rows), 0))
        if fit:
            self.mu_, self.sd_ = num.mean(0), num.std(0) + 1e-9
            if self.categorical:
                self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                    np.array([[str(r[c]) for c in self.categorical] for r in rows], object))
        num = (num - self.mu_) / self.sd_
        if self.categorical:
            cat = self.enc_.transform(np.array([[str(r[c]) for c in self.categorical] for r in rows], object))
            return np.hstack([num, cat])
        return num

    def fit(self, rows: list[dict]) -> "ConditionalLabelGenerator":
        from sklearn.linear_model import LogisticRegression

        self.features.fit(rows)
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000).fit(self._x(rows, fit=True),
                                                                      [str(r[self.label]) for r in rows])
        self.label_values_ = {str(r[self.label]): r[self.label] for r in rows}   # keep original types (bool)
        return self

    def sample(self, n: int) -> list[dict]:
        rows = self.features.sample(n)
        proba = self.clf_.predict_proba(self._x(rows))
        rng = np.random.default_rng(self.seed + 1)
        classes = list(self.clf_.classes_)
        for r, p in zip(rows, proba):
            r[self.label] = self.label_values_[classes[int(rng.choice(len(classes), p=p))]]
        return rows
