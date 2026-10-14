# 17. Rare events and the tail

:::why In one line
Averages are easy. The 0.4% that is fraud, fire, failure or fatality is the entire reason the system exists.
:::

@lab lab42_rare_events

## 17.1 The measurement

A fraud dataset with a 0.4% base rate, three training sets, all scored on the same real gold set:

|Training set|Fraud rate|Gold accuracy|Gold fraud recall|
|---|---|---|---|
|real seed (imbalanced)|0.0035|0.9985|0.933|
|synthetic, generator never produced fraud|0.0000|**cannot even train: one class**|—|
|synthetic + deliberately boosted rare events (5%)|0.0478|0.9998|**1.000**|

A generator that smooths away the tail produces a dataset on which the model cannot learn the thing you care about. And note the trap in column 3: predicting "never fraud" scores about 99.6% accuracy. **Accuracy is the wrong metric for rare events.** Use recall on the rare class, precision at a fixed alert budget, or PR-AUC.

## 17.2 Tail coverage engineering

Generate deliberately: rare events, failure states, boundary conditions, outliers, adversarial cases, long-tail combinations. `perturb.boost_rare()` raises a class to a target rate; `metrics.coverage()` checks the histogram tails are populated.

**Record the original rate.** You boosted the class for learnability; the world did not change. At training time re-weight or re-calibrate, and at evaluation time use the real base rate, or your probability estimates will be badly miscalibrated in production.

## 17.3 Where the tail comes from when you have none

- a simulator you write from domain knowledge (chapter 6.3) — the only unlimited source;
- expert interviews: "describe the five worst incidents you have seen";
- incident reports, post-mortems, regulator bulletins, public recall databases;
- counterfactual perturbation of near-miss real cases.

:::check
- Rare-class recall (not accuracy) is my headline metric.
- Original base rate recorded and used at evaluation.
- Tail coverage measured, not assumed.
:::
