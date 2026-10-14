# 18. Error analysis and active synthetic generation

:::why In one line
Do not generate a million random rows. Find where the model fails, generate data for exactly that, retrain, and measure whether that failure went away.
:::

## 18.1 The loop

```text
evaluate on the REAL gold set
   │
   ├─ collect the failures (worst-scoring items, regressions)
   ├─ label each failure with a CAUSE (coverage gap / wrong label / format / ambiguity / model limit)
   ├─ count the causes: the biggest bucket is your next generation run
   ├─ generate targeted synthetic data for that bucket only
   ├─ retrain
   └─ re-evaluate the SAME failures -> did that bucket improve? did anything else regress?
```

10,000 targeted examples routinely beat 1,000,000 random ones, because the random million repeats what the model already knows.

## 18.2 How to find the failure regions systematically

- **Slice the gold set**: by segment, region, product, severity, length, time. Report the metric per slice; the worst slices are your generation targets.
- **Cluster the errors**: embed the failing inputs and cluster; each cluster is usually one missing capability.
- **Boundary hunting**: where the model's confidence is near 0.5, generate counterfactual pairs across the boundary.
- **Counterfactual probes**: does the decision flip when it should, and stay when it should not (chapter 7.2)?

## 18.3 Keep a regression suite

Every failure you fix becomes a permanent eval item. Over a year this suite, not the generator, is your real asset: it is the only thing preventing the same bug returning in version 12.

## 18.4 When the answer is "more data will not help"

Be honest about the four cases where generation is the wrong tool:

1. The label itself is ambiguous — fix the definition and the annotation guide.
2. The information is not in the input — you need a new feature or a retrieval step, not more rows.
3. The base model cannot do the task at all — change the model.
4. The failure is a product decision, not a modelling error — change the product.

:::check
- Failures labelled by cause before any generation run.
- Targeted batch generated for the biggest bucket only.
- Same failures re-measured after retraining; regressions checked elsewhere.
:::
