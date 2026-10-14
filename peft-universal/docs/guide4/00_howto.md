# 0. How to use this book

Type: **PROJECT + CONCEPT**. You get a working toolkit (`synthkit`) and the method to use it in any domain: finance, healthcare, retail, telecom, manufacturing, cybersecurity, education, SaaS, logistics, government.

## 0.1 The one sentence that matters

> Synthetic data is not "generate more rows". It is: **preserve the real distribution while controlling provenance, diversity, fidelity, privacy, evaluation and failure modes** — and prove it on real data you held back.

## 0.2 The objective. Stop targeting "loss = 0"

Training loss going to zero usually means memorisation. What you actually minimise:

$$ L_total = λ₁·L_task + λ₂·L_distribution + λ₃·L_privacy + λ₄·L_constraint + λ₅·L_bias + λ₆·L_memorization

`synthkit` measures every one of those terms and refuses to ship a dataset that fails a hard gate.

## 0.3 What is in the box

|Module|Does|Chapter|
|---|---|---|
|`synthkit.contract`|the rules your data must obey, written before generating anything|3|
|`synthkit.profile`|measure the real data: marginals, correlations, conditionals, seasonality, text|4|
|`synthkit.generate`|copula (tabular), rule/simulator, LLM, hybrid|6|
|`synthkit.perturb`|calibrated noise, counterfactuals, rare-event boosting, drift|7|
|`synthkit.metrics`|PSI, KL, JS, KS, Wasserstein, TVD, correlation delta, MMD, coverage|8|
|`synthkit.privacy`|DCR, exact match, n-gram overlap, membership inference, PII scan|9|
|`synthkit.judge`|multi-judge, decomposed criteria, human calibration (Cohen's kappa)|10|
|`synthkit.utility`|TSTR: train on synthetic, test on REAL, mixing-ratio sweep|11|
|`synthkit.gates`|10 hard/soft gates + a 6-axis scorecard + ship decision|12|
|`synthkit.forecast`|fit a series generator from real data, backtest on real future|14|
|`synthkit.monitor`|live drift, calibrated thresholds, `L_distribution`, regeneration trigger|15|
|`synthkit.vendors`|where to get the small amount of real data that anchors everything|5|
|`synthkit.pipeline`|all of it, one command|13|

## 0.4 What was executed

:::honest Every number in this book came from a real run in this container, 18 September 2026
- 5 new labs (40-44) executed on CPU. They include results that are **uncomfortable and reported anyway**: a copula generator whose synthetic rows sit too close to real ones (privacy), a drift monitor that cries wolf every month with textbook thresholds, and a "no rare events" dataset that cannot even be trained.
- 13 new tests for `synthkit` (contract, metrics, profile, generator, perturbation, privacy, utility, gates, forecast, monitor, judge, pipeline, vendor catalogue). All pass.
- Two bugs found by those tests and fixed in the code, not hidden: MMD was not exactly zero for identical inputs, and the first generator design distorted marginals badly (PSI 2.2) — replaced by conditional quantile mapping.
- NOT executed: any paid vendor API, any GPU work, and any live production stream. The vendor catalogue lists URLs and licence traps; rate limits are given as *reported ranges* because public sources disagreed when checked, and you must confirm them yourself.
- No price, benchmark or vendor limit in this book is invented.
:::

## 0.5 Prerequisites

Python, and a willingness to hold 20% of your real data back and never look at it. Nothing else.
