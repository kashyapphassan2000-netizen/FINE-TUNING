# 14. Time series and forecasting

@file src/synthkit/forecast.py

## 14.1 Fit the generator to reality, do not invent parameters

`fit_from_profile` estimates level, trend, seasonal amplitude, noise and shock rate from your real series with plain least squares and seasonal averaging — no black box, every line readable. `make_series` then produces as much synthetic history as you want with those parameters.

## 14.2 The test that matters for series

Train the forecaster on **synthetic** history, score it on the **real** future.

@lab lab43_forecast_realtime

In that run, holt-winters fitted on synthetic history scored MAE 34.79 on the real next 14 days versus 46.59 when fitted on the real history itself, and seasonal naive went the other way (28.57 real vs 44.00 synthetic). Read that honestly: on this series the synthetic history was good enough to fit a forecaster, and the method choice mattered more than the data source. Two lessons: always compare against a naive baseline, and always backtest with rolling origins (the lab's three folds give MAE 27.1 / 48.7 / 33.7 — a single split would have told you a comforting lie).

## 14.3 What synthetic series are actually for

- **Cold start**: you have 3 months of data and need to test a forecasting or alerting system across 3 years.
- **Stress testing**: generate the promo spike, the outage, the supply shock, the regime change that has not happened yet.
- **Capacity planning**: replay 10× the volume through your pipeline.
- **Privacy**: share a realistic demand series with a partner without revealing real sales.

What they are **not** for: proving your forecaster is accurate. Only real future data does that.

:::check
- Generator parameters estimated from the real series, not chosen.
- Backtest with rolling origins and a naive baseline.
- Synthetic-trained forecaster scored on real future data.
:::
