# 1. Why data is the leverage point, and what "good" means

## 1.1 Five dimensions, not one

|Dimension|The question|Measured by|
|---|---|---|
|Coverage|are the real cases, including rare ones, present?|`metrics.coverage`, combination coverage|
|Fidelity|is the synthetic distribution close to the real one?|PSI, KS, JS, Wasserstein, TVD, correlation delta, MMD|
|Utility|does training on it improve performance on REAL data?|TSTR ratio (`utility`)|
|Diversity|is the same pattern repeating?|distinct n-grams (text), combination coverage (tabular)|
|Safety|privacy, leakage, toxicity, poisoning?|`privacy`, PII scan, gates|

A dataset can be 1,000,000 rows, 99.9% statistically similar, and still worthless because the 0.1% that matters — the failures, the fraud, the outages — is missing. Chapter 17 shows exactly that failure, measured.

## 1.2 The hierarchy of evidence

1. **Weakest**: "the samples look realistic". Everyone's samples look realistic.
2. **Weak**: statistical similarity on marginals (PSI per column).
3. **Better**: multivariate similarity (correlations, MMD) plus coverage of the tails.
4. **Strong**: privacy tests passed and constraints never violated.
5. **Decisive**: a model trained on the synthetic data performs on a **real, held-out, human-verified** test set.

If you only have time for one thing, do number 5 (chapter 11).

## 1.3 The brutal list

- Synthetic data is not automatically good data. LLM output has style homogeneity, over-clean language, missing edge cases, invented facts and hidden duplication.
- "Our synthetic dataset scored 95% on validation" means nothing if the validation set is also synthetic. That is circular.
- A generator trained on your real data can memorise it. Then your "privacy-safe" dataset is a leak (chapter 9, with the measurement).
- Feeding a model its own output for several generations degrades the distribution (chapter 16, measured).
- More synthetic data is not monotonically better. The mixing ratio is an experiment, not a constant (chapter 11).

:::check
- I can name the five dimensions and how each is measured.
- I accept that the final evidence must come from real data.
:::
