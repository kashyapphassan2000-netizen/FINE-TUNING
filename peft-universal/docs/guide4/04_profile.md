# 4. Step 2: profile the real data

:::why In one line
You cannot mimic what you have not measured. Profiling turns a small real sample into the target your generator must hit and your validator must check against.
:::

@file src/synthkit/profile.py

## 4.1 What it measures

|Group|Statistics|
|---|---|
|numeric|n, mean, std, min, max, q01-q99, skew, excess kurtosis, 3-sigma outlier rate|
|categorical|cardinality, shares, entropy vs max entropy, rare categories (<1%)|
|text|mean/p95 words, vocabulary size, type-token ratio, top words|
|relationships|full correlation matrix, and conditional means for every (numeric \| category) pair|
|temporal|trend per step, ACF at lag 1 / season / 2×season, seasonal strength, shock rate|

The conditional means are the part most people skip. They encode the business logic: "enterprise customers get a response 10× faster". The test suite asserts the profiler recovers exactly that from raw rows.

## 4.2 How little real data you need

Profiling needs **hundreds** of rows, not millions. Lab 40 profiles 360 real rows and generates 2,000 usable synthetic ones. That is the whole reason this approach works for a new business: you need a seed, not a warehouse.

What you cannot get from 300 rows: reliable tail behaviour (rare events), long-horizon seasonality (you need ≥ 2-3 full cycles), and stable estimates for rare categories. Chapter 17 handles the tail deliberately instead of pretending.

## 4.3 Run it

```any
python - <<'PY'
import json
from synthkit.profile import profile, save
rows = [json.loads(l) for l in open("data/real.jsonl")]
p = profile(rows, numeric=["severity","first_response_minutes"],
            categorical=["customer_tier","issue_type"], text=["ticket_text"])
save(p, "reports/profile.json"); print(json.dumps(p["relationships"]["conditional_means"], indent=2))
PY
```

:::warn Profile the TRAIN split only
If you profile all your real data, including the holdout, your "independent" evaluation is contaminated before you start. `synthkit.pipeline` splits 60/20/20 (train / privacy-holdout / gold) and profiles only the first.
:::

:::check
- My profile is saved and versioned next to the contract.
- The conditional means match what a domain expert would tell me.
- I know which statistics my sample is too small to support.
:::
