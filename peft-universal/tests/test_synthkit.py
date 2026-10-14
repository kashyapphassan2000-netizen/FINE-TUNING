import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synthkit import forecast, gates, judge, metrics, monitor, perturb, privacy, profile, utility  # noqa: E402
from synthkit import metrics as _m  # noqa: E402,F401
from synthkit.contract import Contract  # noqa: E402
from synthkit.generate import CopulaGenerator, rule_generator  # noqa: E402
from synthkit.pipeline import run  # noqa: E402
from synthkit.vendors import CATALOG  # noqa: E402

CONTRACT_YAML = """
name: t
version: "1"
fields:
  - {name: tier, dtype: category, categories: [free, pro]}
  - {name: sev, dtype: int, min: 1, max: 5}
  - {name: mins, dtype: float, min: 0}
  - {name: churned, dtype: bool}
constraints: ["mins >= 0", "sev <= 5"]
privacy: {max_exact_match_rate: 0.0, min_dcr_ratio: 0.5}
quality: {max_psi: 0.2, min_utility_ratio: 0.8, max_duplicate_rate: 0.05}
"""


def fake_real(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        tier = str(rng.choice(["free", "pro"], p=[0.7, 0.3]))
        sev = int(np.clip(rng.normal(3 if tier == "free" else 2, 1), 1, 5))
        mins = float(max(0.1, rng.gamma(2, 40 if tier == "free" else 15)))
        churn = bool(rng.random() < 1 / (1 + np.exp(-(-2 + 0.6 * sev + 0.003 * mins))))
        rows.append({"tier": tier, "sev": sev, "mins": round(mins, 2), "churned": churn})
    return rows


@pytest.fixture(scope="module")
def contract(tmp_path_factory):
    p = tmp_path_factory.mktemp("c") / "c.yaml"
    p.write_text(CONTRACT_YAML)
    return Contract.load(p)


def test_contract_catches_every_violation_type(contract):
    bad = [{"tier": "enterprise", "sev": 9, "mins": -5.0, "churned": "yes"},
           {"tier": "free", "sev": 2, "churned": True},                      # missing column
           {"tier": "pro", "sev": 3, "mins": 10.0, "churned": False}]        # good
    r = contract.validate(bad)
    v = r["violations"]
    assert v["unknown_category:tier"] == 1 and v["above_max:sev"] == 1
    assert v["below_min:mins"] == 1 and v["not_bool:churned"] == 1
    assert v["missing_column:mins"] == 1 and v["constraint:sev <= 5"] == 1
    assert not r["ok"]
    assert contract.validate([bad[2]])["ok"]


def test_metrics_behave_like_the_textbook():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)
    shifted = rng.normal(1.5, 1, 4000)
    assert metrics.psi(a, b) < 0.05 < metrics.psi(a, shifted)
    assert metrics.ks_statistic(a, b) < metrics.ks_statistic(a, shifted)
    assert metrics.js_divergence(a, b) < metrics.js_divergence(a, shifted)
    assert metrics.wasserstein1(a, shifted) == pytest.approx(1.5, abs=0.15)
    assert metrics.categorical_tvd(["a"] * 7 + ["b"] * 3, ["a"] * 5 + ["b"] * 5) == pytest.approx(0.2, abs=0.01)
    assert metrics.coverage(a, a) == 1.0
    assert metrics.coverage(a, a[a > 0]) < 0.6           # missing the left tail
    assert metrics.mmd_rbf(np.c_[a, b], np.c_[a, b]) == pytest.approx(0.0, abs=1e-6)
    rows_real = [{"x": 1.0, "c": "a"}, {"x": 2.0, "c": "b"}]
    rep = metrics.table_report(rows_real, rows_real, ["x"], ["c"])
    assert rep["summary"]["verdict"] == "stable"
    assert metrics.combination_coverage(rows_real, [{"c": "a"}], ["c"]) == 0.5


def test_profile_captures_shape_and_relationships():
    rows = fake_real(500)
    p = profile.profile(rows, ["sev", "mins"], ["tier", "churned"])
    assert p["n_rows"] == 500 and p["numeric"]["mins"]["q95"] > p["numeric"]["mins"]["median"]
    assert p["numeric"]["mins"]["skew"] > 0                      # gamma is right-skewed
    assert set(p["categorical"]["tier"]["shares"]) == {"free", "pro"}
    cond = p["relationships"]["conditional_means"]["mins|tier"]
    assert cond["free"] > cond["pro"]                            # the real structure is detected
    t = profile.profile([{"t": i, "v": float(np.sin(i / 7 * 2 * np.pi) * 10 + 100)} for i in range(120)],
                        ["v"], [], time_col="t", value_col="v", season=7)
    assert t["temporal"]["seasonal_strength"] > 0.5


def test_copula_generator_preserves_marginals_and_correlation():
    real = fake_real(800)
    gen = CopulaGenerator(numeric=["sev", "mins"], categorical=["tier", "churned"], seed=0).fit(real)
    synth = gen.sample(800)
    assert len(synth) == 800 and set(synth[0]) == {"sev", "mins", "tier", "churned"}
    rep = metrics.table_report(real, synth, ["sev", "mins"], ["tier", "churned"])
    assert rep["summary"]["max_psi"] < 0.2
    assert rep["multivariate"]["correlation_delta"] < 0.25
    assert rep["categorical"]["tier"]["synth_shares"]["free"] == pytest.approx(0.7, abs=0.08)
    sim = rule_generator(50, lambda rng, i: {"i": i, "v": float(rng.normal())}, seed=1)
    assert len(sim) == 50 and sim[0]["i"] == 0


def test_perturbations_are_calibrated():
    rows = [{"x": 10.0, "y": "a", "id": i} for i in range(1000)]
    noisy = perturb.feature_noise(rows, ["x"], rel_sigma=0.1, seed=0)
    assert np.std([r["x"] for r in noisy]) == pytest.approx(1.0, abs=0.15)
    flipped = perturb.label_noise(rows, "y", 0.2, ["a", "b"], seed=0)
    assert np.mean([r["y"] != "a" for r in flipped]) == pytest.approx(0.2, abs=0.04)
    miss = perturb.missingness(rows, ["x"], 0.3, seed=0)
    assert np.mean([r["x"] is None for r in miss]) == pytest.approx(0.3, abs=0.05)
    cf = perturb.counterfactual(rows[:5], "y", "b")
    assert all(c["y"] == "b" and c["_counterfactual_of"] == c["_counterfactual_of"] for c in cf)
    rng_gen = lambda rg: {"x": 1.0, "y": "rare", "id": -1}  # noqa: E731
    boosted = perturb.boost_rare(rows, lambda r: r["y"] == "rare", 0.1, rng_gen, seed=0)
    assert np.mean([r["y"] == "rare" for r in boosted]) == pytest.approx(0.1, abs=0.02)
    assert perturb.adversarial_text([{"t": "hi"}], "t", "contradiction")[0]["t"].endswith("happened.")
    assert perturb.drift([{"v": 100.0}], "v", 0.2)[0]["v"] == pytest.approx(120.0)


def test_privacy_detects_copying_and_clears_independent_data():
    real = fake_real(300, seed=1)
    holdout = fake_real(300, seed=2)
    copied = [dict(r) for r in real[:200]]                       # worst case: the generator memorised
    indep = CopulaGenerator(["sev", "mins"], ["tier", "churned"], seed=3).fit(real).sample(300)
    cols = ["sev", "mins"]
    assert privacy.exact_match_rate(real, copied, cols) == 1.0
    assert privacy.exact_match_rate(real, indep, ["sev", "mins", "tier"]) < 0.5
    assert privacy.dcr(real, copied, cols)["ratio"] == pytest.approx(0.0, abs=1e-6)
    # measured property: resampling real values keeps synthetic rows close to real ones...
    assert privacy.dcr(real, indep, cols)["ratio"] < 0.3
    smoothed = CopulaGenerator(["sev", "mins"], ["tier", "churned"], seed=3, smoothing=0.25).fit(real).sample(300)
    # ...and jittering the draws fixes it without wrecking fidelity
    assert privacy.dcr(real, smoothed, cols)["ratio"] > privacy.dcr(real, indep, cols)["ratio"]
    assert privacy.dcr(real, smoothed, cols)["ratio"] > 0.5
    assert metrics.table_report(real, smoothed, ["mins"], [])["summary"]["max_psi"] < 0.25
    assert privacy.membership_inference_auc(real, holdout, copied, cols) > 0.6
    assert privacy.scan_pii([{"t": "mail me at a.b@c.com"}], ["t"]) == {"email": 1}
    assert privacy.ngram_overlap(["the quick brown fox jumps over the lazy dog today"],
                                 ["the quick brown fox jumps over the lazy dog today"]) == 1.0


def test_utility_tstr_and_ratio():
    real = fake_real(600, seed=4)
    train, gold = real[:400], real[400:]
    synth = CopulaGenerator(["sev", "mins"], ["tier", "churned"], seed=5).fit(train).sample(800)
    r_only = utility.train_test(train, gold, ["sev", "mins"], ["tier"], "churned")
    s_only = utility.train_test(synth, gold, ["sev", "mins"], ["tier"], "churned")
    assert 0.0 <= r_only["accuracy"] <= 1.0 and "auc" in r_only
    assert utility.utility_ratio(r_only, s_only) > 0.7
    mixes = utility.tstr(train, gold, synth, ["sev", "mins"], ["tier"], "churned", ratios=(0.0, 0.5, 1.0))
    assert len(mixes) == 3 and mixes[0]["synthetic_share"] == 0.0 and mixes[-1]["n_real"] == 0


def test_gates_reject_on_hard_failures(contract):
    dist = {"summary": {"max_psi": 0.05}, "numeric": {"x": {"coverage": 1.0}}}
    ok_priv = {"dcr": {"ratio": 1.0}, "exact_match_rate": 0.0, "membership_inference_auc": 0.5, "pii_hits": {}}
    good = gates.run_gates({"ok": True, "violations": {}}, dist, ok_priv, {"ratio": 1.0}, 0.0, 1.0,
                           contract=contract)
    assert good["decision"] == "SHIP" and good["scorecard"]["privacy"] == 100
    leaky = {**ok_priv, "exact_match_rate": 0.4, "membership_inference_auc": 0.9, "pii_hits": {"email": 3}}
    bad = gates.run_gates({"ok": True, "violations": {}}, dist, leaky, {"ratio": 0.5}, 0.3, 0.2,
                          contract=contract)
    assert bad["decision"] == "REJECT"
    assert {"4 privacy: exact match", "6 privacy: MIA AUC", "7 PII", "10 utility (TSTR/TRTR)"} <= set(bad["hard_failures"])
    assert bad["scorecard"]["privacy"] < 20     # 1.0 ratio x (1-0.4 exact) x (1-2x0.4 MIA) = 0.12


def test_forecast_and_backtest():
    rng = np.random.default_rng(0)
    t = np.arange(200)
    series = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, 200)
    p = forecast.fit_from_profile(series[:150], season=7)
    assert p["trend"] == pytest.approx(0.5, abs=0.1) and p["season_amp"] > 10
    synth = forecast.make_series(150, **{k: v for k, v in p.items() if k != "seed"}, seed=1)
    assert abs(float(np.mean(synth)) - float(np.mean(series[:150]))) < 25
    hw = forecast.holt_winters(series[:150], 14, 7)
    sn = forecast.seasonal_naive(series[:150], 14, 7)
    assert forecast.metrics(series[150:164], hw)["mae"] < 3 * forecast.metrics(series[150:164], sn)["mae"]
    bt = forecast.backtest(series, horizon=14, season=7, folds=3)
    assert len(bt["folds"]) == 3 and bt["mean_mae"] > 0


def test_monitor_calibration_beats_textbook_thresholds():
    rng = np.random.default_rng(0)
    quiet = [{"v": float(x), "c": "a"} for x in rng.normal(100, 10, 400)]
    event = [{"v": float(x), "c": "a"} for x in rng.normal(140, 10, 100)]
    cal = monitor.calibrate_thresholds(quiet, ["v"], ["c"], window=100)
    assert cal["n_windows"] >= 2 and cal["act"] >= cal["warn"] >= cal["median_quiet_drift"]
    m = monitor.DriftMonitor(quiet[:100], ["v"], ["c"], warn=cal["warn"], act=cal["act"])
    assert m.score(quiet[100:200], "quiet")["status"] == "STABLE"
    assert m.score(event, "event")["status"] == "ACT"
    loss = monitor.synthetic_vs_real_loss(quiet, event, ["v"], ["c"])
    assert loss["L_distribution"] > 0 and set(loss["parts"]) == {"psi", "ks", "tvd"}


def test_judge_scoring_and_calibration():
    perfect = {"correctness": 4, "completeness": 4, "instruction": 4, "safety": 4, "style": 4}
    assert judge.weighted(perfect) == 1.0
    assert judge.weighted({**perfect, "safety": 0}) == pytest.approx(0.85, abs=0.01)
    a = judge.agreement([0.9, 0.8, 0.4, 0.3], [1, 1, 0, 0])
    assert a["agreement"] == 1.0 and a["cohen_kappa"] == 1.0
    bad = judge.agreement([0.9, 0.9, 0.9, 0.9], [1, 1, 0, 0])
    assert bad["agreement"] == 0.5 and bad["false_keeps"] == 2
    best = judge.best_threshold([0.6, 0.55, 0.2, 0.1], [1, 1, 0, 0])
    assert best["cohen_kappa"] == 1.0


def test_pipeline_end_to_end(contract, tmp_path):
    real = fake_real(300, seed=7)
    res = run(real, contract, ["sev", "mins"], ["tier"], "churned", n=500, seed=0)
    assert len(res["synth"]) == 500
    assert res["contract_report"]["ok"], res["contract_report"]["violations"]
    assert res["distribution"]["summary"]["max_psi"] < 0.3
    assert res["gates"]["decision"] in ("SHIP", "SHIP WITH WARNINGS", "REJECT")
    assert set(res["gates"]["scorecard"]) == {"fidelity", "coverage", "diversity", "utility", "privacy", "cleanliness"}
    assert res["privacy"]["membership_inference_auc"] <= 1.0
    assert json.dumps(res["utility"]["mixtures"])


def test_vendor_catalog_is_honest():
    assert len(CATALOG) >= 15
    assert all(s.url.startswith("http") or s.url == "internal" for s in CATALOG)
    om = next(s for s in CATALOG if "Open-Meteo" in s.name)
    assert "NON-commercial" in om.commercial_use          # the licence trap is recorded, not hidden
    assert any("your own product logs" in s.name.lower() for s in CATALOG)
