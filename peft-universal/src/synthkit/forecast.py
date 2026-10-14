"""Time series: generate synthetic series that look like the real one, and prove it by forecasting.

The test that matters: train a forecaster on synthetic history, evaluate on REAL future. If it holds
up, your synthetic generator captured the dynamics, not just the average.
"""
from __future__ import annotations

import numpy as np


def make_series(n: int, level: float = 100.0, trend: float = 0.05, season: int = 7,
                season_amp: float = 10.0, noise: float = 3.0, shock_rate: float = 0.01,
                shock_size: float = 40.0, seed: int = 0) -> np.ndarray:
    """A transparent generator: level + trend + seasonality + noise + rare shocks. Fit its parameters
    to your real profile (synthkit.profile temporal block) instead of guessing them."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = level + trend * t + season_amp * np.sin(2 * np.pi * t / season)
    y = base + rng.normal(0, noise, n)
    shocks = rng.random(n) < shock_rate
    y[shocks] += rng.choice([-1, 1], shocks.sum()) * shock_size
    return np.maximum(y, 0)


def fit_from_profile(real: np.ndarray, season: int = 7, seed: int = 0) -> dict:
    """Estimate the generator's parameters from a real series (no libraries, all transparent)."""
    t = np.arange(len(real))
    slope, intercept = np.polyfit(t, real, 1)
    detrended = real - (slope * t + intercept)
    seasonal = np.array([detrended[i::season].mean() for i in range(season)])
    resid = detrended - np.tile(seasonal, int(np.ceil(len(real) / season)))[:len(real)]
    sd = float(resid.std())
    return {"level": float(intercept), "trend": float(slope), "season": season,
            "season_amp": float(np.abs(seasonal).max()), "noise": sd,
            "shock_rate": float(np.mean(np.abs(resid) > 3 * sd)), "shock_size": float(3.5 * sd), "seed": seed}


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    return np.array([history[-season + (i % season)] for i in range(horizon)])


def holt_winters(history: np.ndarray, horizon: int, season: int = 7, alpha: float = 0.3,
                 beta: float = 0.1, gamma: float = 0.3) -> np.ndarray:
    """Additive Holt-Winters, written out so you can see every line (no statsmodels dependency)."""
    y = np.asarray(history, float)
    level = y[:season].mean()
    trend = (y[season:2 * season].mean() - level) / season if len(y) >= 2 * season else 0.0
    seas = list(y[:season] - level)
    for i in range(len(y)):
        s = seas[i % season]
        prev = level
        level = alpha * (y[i] - s) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
        seas[i % season] = gamma * (y[i] - level) + (1 - gamma) * s
    return np.array([level + (h + 1) * trend + seas[(len(y) + h) % season] for h in range(horizon)])


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    smape = float(np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p) + 1e-9)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "smape": round(smape, 4)}


def backtest(series: np.ndarray, horizon: int, season: int = 7, folds: int = 3, method=holt_winters) -> dict:
    """Rolling-origin evaluation. One train/test split on a time series is not evidence."""
    res = []
    for f in range(folds):
        cut = len(series) - horizon * (folds - f)
        if cut <= 2 * season:
            continue
        pred = method(series[:cut], horizon, season)
        res.append(metrics(series[cut:cut + horizon], pred))
    return {"folds": res, "mean_mae": round(float(np.mean([r["mae"] for r in res])), 4),
            "mean_smape": round(float(np.mean([r["smape"] for r in res])), 4)}
