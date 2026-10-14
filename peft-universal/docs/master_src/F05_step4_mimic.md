# 7. Step 4 — Make the synthetic data mimic reality (measure the gap, then close it)

:::why The requirement, stated properly
"99.9% like the real data" is not a claim you can make by looking at samples. It is a **measured distance** between two distributions, driven down by a loop.
:::

## 7.1 The loop

```text
REAL SAMPLE (even a few hundred rows)
   └─> PROFILE           marginals, correlations, conditional structure, seasonality, rare events
        └─> GENERATE     copula / simulator / LLM / hybrid, constrained by a data contract
             └─> MEASURE THE GAP   PSI, KS, JS, Wasserstein, TVD, correlation delta, MMD, coverage
                  └─> FIX the worst column or relationship
                       └─> repeat until the gap is inside your threshold
                            └─> PRIVACY + UTILITY gates, then ship the dataset
```

## 7.2 The commands

```any
python -m synthkit.pipeline --real data/real.jsonl --contract contracts/domain.yaml \
    --numeric <cols> --categorical <cols> --label <target> --n 5000 --smoothing 0.25 --out reports/synth
python -m synthkit.vendors --domain <your domain> --free-only        # where to get a real anchor sample
```

## 7.3 What "close enough" means, in numbers

|Metric|Reading|
|---|---|
|PSI per column|< 0.1 stable · 0.1-0.25 investigate · > 0.25 large shift **(calibrate on your own quiet data)**|
|TVD (categories)|share difference; < 0.05 is usually fine|
|correlation delta|mean absolute difference of correlation matrices — relationships, not just marginals|
|coverage|are the real histogram bins, including the tails, populated?|
|**utility (TSTR)**|train on synthetic, test on REAL: the only decisive number|
|privacy (DCR, exact match, membership AUC)|synthetic rows must not BE real rows|

Two measured warnings from the deep chapters: a generator can score well on fidelity and still **fail privacy** (synthetic rows sitting on real ones — a one-parameter fix flipped the decision from REJECT to SHIP), and recursive training on your own generated data **drifts** the distribution unless you keep real data in every round.

## 7.4 Rare events are the point

Averages are easy; the 0.4% that is fraud, failure or an outage is why the system exists. Generate the tail deliberately, record the original base rate, and re-weight at training time. A dataset with no rare events produced a model that could not be trained on them at all.

:::check
- Gap measured per column AND multivariate, not eyeballed.
- Tail coverage checked; rare classes deliberately generated.
- Utility on a REAL holdout beats real-only or comes close, and privacy gates pass.
:::
