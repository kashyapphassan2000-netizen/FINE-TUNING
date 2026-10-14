# 15. Real time: comparing live data with your synthetic assumption

:::why In one line
The world moves. Your synthetic dataset was built from a snapshot. A monitor tells you when the snapshot is stale, and the loop regenerates before your model quietly degrades.
:::

@file src/synthkit/monitor.py

## 15.1 The loop

```text
live real batch ──> DriftMonitor.score() ──> STABLE  : keep serving
                                        ──> INVESTIGATE : check the source, look at the columns
                                        ──> ACT   : re-profile real -> regenerate -> retrain
                                                     -> evaluate on REAL holdout -> then move the reference
```

Never move the reference window before you have retrained and evaluated. Otherwise the monitor silently accepts the new reality and you lose the signal.

## 15.2 The measured lesson: textbook thresholds cry wolf

@lab lab43_forecast_realtime

Three configurations of the same monitor on the same data:

|Configuration|Result|
|---|---|
|fixed reference, textbook thresholds (0.1 / 0.25)|**every** month says ACT — the series simply trends upward, so the reference goes stale by itself|
|rolling 30-day reference, textbook thresholds|still ACT on ordinary months: PSI on 30-point windows is noisy|
|rolling reference + thresholds calibrated on quiet history|ordinary months STABLE (0.49, 0.50), the +25% price cut fires ACT (12.43), the return to normal is STABLE|

`calibrate_thresholds()` measures the drift distribution across your own quiet periods and sets warn/act at its high quantiles. **Calibrate on your data; the 0.1/0.25 rule is a starting point, not a law.** A monitor that fires every month gets muted, and then misses the real event.

## 15.3 The number that replaces "loss = 0"

`synthetic_vs_real_loss()` returns a weighted distribution gap (`L_distribution`) from PSI, KS and TVD. Track it per dataset release alongside utility. It is the scalar you report to a stakeholder who asks "how close is our synthetic data to reality this quarter?"

## 15.4 What to monitor in production

|Signal|Why|
|---|---|
|input drift per feature|the world changed|
|category share drift|new products, new regions|
|label/outcome drift|the thing you predict changed|
|model score distribution|silent degradation before metrics move|
|utility on a fresh real slice|the only ground truth; schedule it monthly|

:::check
- Thresholds calibrated on my own quiet history.
- Reference window rolls, and only moves after retraining.
- `L_distribution` tracked per release.
:::
