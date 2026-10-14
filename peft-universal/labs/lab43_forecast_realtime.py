"""Lab 43: fit a synthetic series to a real one, forecast with it, then run the real-time drift loop."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthkit.forecast import backtest, fit_from_profile, holt_winters, make_series, metrics, seasonal_naive  # noqa: E402
from synthkit.monitor import DriftMonitor, calibrate_thresholds, synthetic_vs_real_loss  # noqa: E402

rng = np.random.default_rng(0)
# "REAL" demand: trend + weekly season + noise + rare promo spikes (a stand-in for your sales data)
t = np.arange(240)
real = 500 + 0.8 * t + 60 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 25, 240)
real[rng.random(240) < 0.03] += 300
real = np.maximum(real, 0)

params = fit_from_profile(real[:180], season=7)
print("fitted generator parameters from 180 real days (no libraries, all estimated from the data):")
print("  " + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in params.items()))
synth = make_series(n=180, **{k: v for k, v in params.items() if k != "seed"}, seed=7)
print(f"\nreal   mean {real[:180].mean():8.1f} std {real[:180].std():7.1f} "
      f"min {real[:180].min():7.1f} max {real[:180].max():7.1f}")
print(f"synth  mean {synth.mean():8.1f} std {synth.std():7.1f} min {synth.min():7.1f} max {synth.max():7.1f}")

print("\nA) FORECAST QUALITY: train the forecaster on real vs on synthetic, always score on REAL future")
horizon = 14
future = real[180:180 + horizon]
for name, hist in [("real history", real[:180]), ("synthetic history", synth)]:
    for method, label in [(seasonal_naive, "seasonal naive"), (holt_winters, "holt-winters")]:
        pred = method(hist, horizon, 7)
        m = metrics(future, pred)
        print(f"  {name:<20}{label:<16} MAE {m['mae']:8.2f}  RMSE {m['rmse']:8.2f}  sMAPE {m['smape']:.3f}")
print("  (a forecaster fitted on synthetic history is only useful if its REAL-future error is close)")

print("\nB) BACKTEST on the real series (rolling origin, 3 folds):", backtest(real, horizon=14, season=7))

def rows(lo, hi, shift=1.0):
    return [{"demand": float(real[i] * shift), "weekday": str(int(i % 7)), "promo": bool(real[i] * shift > 900)}
            for i in range(lo, hi)]


print("\nC) REAL-TIME LOOP: score each new month of real data against a reference window")
print("   C1: FIXED reference (days 0-120). Note what happens even when nothing unusual occurs:")
fixed = DriftMonitor(rows(0, 120), numeric=["demand"], categorical=["weekday", "promo"])
rolling = DriftMonitor(rows(90, 120), numeric=["demand"], categorical=["weekday", "promo"])
batches = [(120, 150, 1.0, "ordinary month"), (150, 180, 1.0, "ordinary month"),
           (180, 210, 1.25, "price cut: +25%"), (210, 240, 1.0, "back to normal")]
print(f"{'batch':<22}{'worst':>9}{'status':>14}")
for lo, hi, shift, note in batches:
    r = fixed.score(rows(lo, hi, shift), note)
    print(f"{note:<22}{r['worst']:>9}{r['status']:>14}")
print("   The series has an upward trend of ~0.8/day, so a FIXED reference goes stale by itself.")
print("   A monitor that cries ACT every month gets ignored, and then misses the real event.")

print("\n   C2: ROLLING reference (previous 30 days), updated after each batch is accepted:")
print(f"{'batch':<22}{'worst':>9}{'status':>14}   action")
for lo, hi, shift, note in batches:
    b = rows(lo, hi, shift)
    r = rolling.score(b, note)
    print(f"{note:<22}{r['worst']:>9}{r['status']:>14}   {r['action'][:46]}")
    if r["status"] != "ACT":
        rolling.update_reference(b)      # only adopt reality as the new baseline when it is not an incident
print("   Still noisy: PSI on 30-point windows moves a lot by chance, so textbook 0.1/0.25 fires constantly.")

print("\n   C3: CALIBRATED thresholds, measured on the quiet history itself:")
cal = calibrate_thresholds(rows(0, 120), ["demand"], ["weekday", "promo"], window=30)
print("   ", cal)
rolling2 = DriftMonitor(rows(90, 120), ["demand"], ["weekday", "promo"], warn=cal["warn"], act=cal["act"])
print(f"{'batch':<22}{'worst':>9}{'status':>14}   action")
for lo, hi, shift, note in batches:
    b = rows(lo, hi, shift)
    r = rolling2.score(b, note)
    print(f"{note:<22}{r['worst']:>9}{r['status']:>14}   {r['action'][:44]}")
    if r["status"] != "ACT":
        rolling2.update_reference(b)
print("   Now only the real event fires. Calibrate alert thresholds on your own quiet periods, always.")

print("\nD) THE NUMBER TO TRACK instead of 'training loss = 0':")
synth_rows = [{"demand": float(x), "weekday": str(int(i % 7)), "promo": bool(x > 900)} for i, x in enumerate(synth)]
print("  ", synthetic_vs_real_loss(rows(0, 120), synth_rows, ["demand"], ["weekday", "promo"]))
print("\nWhen L_distribution rises or the monitor says ACT: re-profile the real data, regenerate, retrain,")
print("re-evaluate on a REAL holdout, and only then move the monitor's reference window forward.")
