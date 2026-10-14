# 2. The architecture

```text the closed loop (everything in this book is one of these boxes)
REAL / LEGAL DATA
   │
   ├─[1] data contract .................... synthkit.contract      (ch 3)
   ├─[2] schema + semantic profiling ...... synthkit.profile       (ch 4)
   ├─[3] privacy / PII / IP / compliance .. synthkit.privacy       (ch 9)
   ├─[4] real distribution estimation ..... profile + forecast     (ch 4, 14)
   │        marginals · correlations · conditionals · seasonality · rare events · business rules
   ├─[5] generation ....................... synthkit.generate      (ch 6)
   │        statistical · ML · LLM · simulator · hybrid
   ├─[6] noise / perturbation / counterfactuals .. synthkit.perturb (ch 7)
   ├─[7] validation ....................... synthkit.metrics       (ch 8)
   │        schema · statistical · semantic · utility · privacy · constraints
   ├─[8] filtering / dedup / hard negatives ..... metrics + gates  (ch 8, 12)
   ├─[9] human + LLM + programmatic judging ..... synthkit.judge   (ch 10)
   ├─[10] train ................................ your model
   ├─[11] evaluate against REAL holdout ........ synthkit.utility  (ch 11)
   ├─[12] error analysis ....................... ch 18
   └─[13] regenerate TARGETED synthetic data ... ch 18 (active generation)
            └──────────── loop, with versioned lineage ────────────┘
```

Without steps 11-13 you do not have a synthetic data system; you have a text generator.

## 2.1 Where each piece runs

|Stage|Machine|Cost driver|
|---|---|---|
|profiling, contracts|laptop|free|
|tabular generation (copula, simulator)|laptop|free, seconds for 100k rows|
|LLM generation / judging|API or your own vLLM|tokens: the main bill|
|validation, privacy, gates|laptop|free|
|utility (TSTR)|laptop (sklearn) or GPU (LLM fine-tune)|minutes to hours|
|monitoring|wherever your data lands|cheap, but must be scheduled|

## 2.2 Honest limitations of this toolkit

- The tabular generator is a Gaussian copula with conditional quantile mapping. It captures marginals, correlations and per-group structure. It does **not** capture long chains of causal logic, multi-table foreign keys or sequence dynamics — for those, write a simulator (chapter 6.3) or use a time-series generator (chapter 14).
- No differential privacy. `synthkit.privacy` **measures** leakage; it does not provide a formal guarantee. Regulated data needs a DP generator and a lawyer, in that order.
- The LLM stages need an OpenAI-compatible endpoint. The labs run against a local fake server so everything is reproducible without a key.
- Nothing here has been run against a production stream. The monitor is tested on replayed data.
