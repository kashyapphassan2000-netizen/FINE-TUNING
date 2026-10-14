# 7. Step 4: noise, counterfactuals and drift — calibrated, not random

@file src/synthkit/perturb.py

## 7.1 Why deliberately degrade your own data

Real data is messy. A model trained only on clean synthetic data meets typos, missing fields, contradictory instructions and sensor drift on day one and falls over. You add these **on purpose**, in controlled amounts, matched to the error rates you measured in the real sample.

|Kind|Function|Calibrate against|
|---|---|---|
|feature noise|`feature_noise(rel_sigma)`|real measurement/sensor error|
|label noise|`label_noise(rate)`|your annotators' real disagreement rate|
|missingness|`missingness(rate)`|real missing-value rates per column|
|semantic/adversarial|`adversarial_text(kind)`|real user behaviour: contradictions, no context, irrelevance|
|distribution shift|`drift(col, shift)`|the change you expect next quarter|
|counterfactual|`counterfactual(factor, value)`|the fairness/causal question you must answer|

The tests assert each one produces the requested rate (label flips 0.2 ± 0.04, missingness 0.3 ± 0.05, noise std 1.0 ± 0.15 on a mean-10 column at 10%).

## 7.2 Counterfactuals: the enterprise superpower

Change exactly one factor, hold everything else, and ask whether the decision changes:

```text
real:           income 80k, credit_score 760 -> approved
counterfactual: income 80k, credit_score 620 -> should flip to declined
counterfactual: income 80k, credit_score 760, different postcode -> must NOT change
```

That gives you fairness tests, causal robustness tests and regression tests that no amount of ordinary data provides. Generate them systematically for every protected attribute and every decision boundary.

## 7.3 The rule about noise

Noise must be calibrated against **real** error distributions. Arbitrary noise makes the dataset harder without making the model better, and it corrupts your fidelity metrics. Measure first (chapter 4's outlier rate and missingness), then inject.

:::check
- Every noise rate traces to a measured real rate.
- Counterfactual sets exist for each protected attribute and decision boundary.
- Noised rows are flagged (`_label_noised`, `_adversarial`) so you can exclude them from fidelity checks.
:::
