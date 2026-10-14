# 9. Step 6: privacy, memorization and leakage

:::why In one line
A synthetic row that is a real row is a data breach with extra steps. Measure it every batch.
:::

@file src/synthkit/privacy.py

## 9.1 The four tests

|Test|What it catches|Healthy value|
|---|---|---|
|exact match rate|the generator copied whole records|0|
|DCR (distance to closest record) ratio|synthetic rows sitting on top of real ones|≥ ~1 (synth-to-real distances similar to real-to-real)|
|membership inference AUC|"was this record in the training data?"|≈ 0.5 (0.5 = no signal)|
|n-gram overlap (text)|copied sentences|near 0 for 8-grams|
|PII scan|emails, phones, cards, IDs, IPs in output|0 hits|

## 9.2 The uncomfortable measurement

The tests plant a memorising generator (synthetic = copies of real rows) and confirm: exact match 1.0, DCR ratio 0.0, membership-inference AUC above 0.6. Then they check an honest copula generator and find **DCR ratio 0.17** — better, but still low, because empirical-quantile sampling reuses real values. `smoothing=0.25` raises it above 0.5 while PSI stays under 0.25.

That is the honest state of affairs: **a generator that resamples your data carries some of it.** Measure, do not assume.

## 9.3 What this module is not

It measures leakage; it does not guarantee its absence. For regulated data you need, in order: a lawyer, a lawful basis, and a differentially private generator (DP-SGD-trained model or a DP synthesiser) with an agreed ε. `synthkit.privacy` then becomes your monitoring, not your defence. There is no DP implementation in this toolkit, and pretending otherwise would be dangerous.

## 9.4 Small datasets are the risky ones

The smaller the real seed, the more a generator memorises it. With 200 rows, treat every synthetic record as potentially identifying until the four tests say otherwise, and never publish such a dataset externally without a DP guarantee and a review.

:::check
- All four tests run on every batch, not on the first one only.
- Thresholds are in the contract (`min_dcr_ratio`, `max_exact_match_rate`).
- Regulated data: DP generator and legal sign-off, not just these metrics.
:::
