# 8. Step 5: validation — is it actually like the real thing?

@file src/synthkit/metrics.py

## 8.1 The metrics, and how to read them

|Metric|Question|Rule of thumb|
|---|---|---|
|PSI|has this column's distribution shifted?|<0.1 stable, 0.1-0.25 investigate, >0.25 large shift **(calibrate on your own data — see ch 15)**|
|KS|biggest gap between the two CDFs|0 identical, 1 disjoint|
|JS divergence|symmetric distribution distance|0 identical|
|Wasserstein-1|average "distance to move" between distributions|in the column's own units: interpretable|
|TVD (categorical)|difference in category shares|0.5·Σ\|p−q\||
|correlation delta|are the relationships preserved?|mean absolute difference of correlation matrices|
|MMD (RBF)|multivariate: same joint distribution?|0 identical (the tests assert exactly 0 for identical input)|
|coverage|does synthetic populate the real histogram bins, including tails?|1.0 = every real bin represented|
|combination coverage|which real category combinations exist in synthetic?|tabular diversity measure|
|distinct n-grams|is the text repetitive?|text diversity measure|

## 8.2 Measured on a real pipeline

@lab lab40_end_to_end

Read that output in order: the contract passes, the marginals are close (PSI 0.01 and 0.06), the categorical shares match (TVD 0.02, 0.008), the correlation structure is preserved, privacy is measured, and then the only question that matters — utility on the real gold holdout — is answered.

:::warn Two metric traps this book hit and fixed
- **Wrong metric for the data type.** Distinct 3-grams on two-word category strings returns 0.0 and the gate fails for no reason. Tabular diversity needs combination coverage, not n-grams.
- **Duplicates are not always a defect.** With five severity levels and four issue types, repeats are inevitable. Judge the **excess** over the real data's own duplicate rate, not the raw number. The pipeline reports synthetic, real baseline and excess separately.
:::

## 8.3 Filtering, dedup and hard negatives

After measuring: drop exact and near duplicates, drop constraint violations, drop records the judge rejects (chapter 10), and keep a deliberate set of **hard negatives** — near-miss cases that look like the positive class. A dataset of easy examples produces a model that is confident and wrong at the boundary.

:::check
- Fidelity measured per column AND multivariate.
- Metrics chosen for the data type.
- Duplicate rate compared against the real baseline.
:::
