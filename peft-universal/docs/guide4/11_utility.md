# 11. Step 8: utility — the only experiment that settles it

:::why In one line
Train on synthetic, test on REAL. If the model does not get better on data it has never seen, nothing else you measured matters.
:::

@file src/synthkit/utility.py

## 11.1 The five models to train

|Model|Training data|Tells you|
|---|---|---|
|1|real only|the baseline you must beat (TRTR)|
|2|synthetic only|whether synthetic can stand alone (TSTR)|
|3|real + synthetic|the usual production answer|
|4|real + **high-quality** synthetic (judge-filtered)|whether filtering pays|
|5|real + targeted hard examples|whether active generation pays (chapter 18)|

All five are evaluated on the same **gold test set**: real, held out, ideally human-verified, never used for generation or tuning.

$$ utility ratio = metric(TSTR) / metric(TRTR)     (1.0 = synthetic is as useful as real)

## 11.2 Measured

@lab lab40_end_to_end

In that run: real-only accuracy on the gold holdout, synthetic-only accuracy, and the mixing sweep from 0% to 100% synthetic. The utility ratio is reported by the pipeline and is gate 10 — a hard gate, because a dataset that does not help is not worth its privacy risk.

## 11.3 The mixing ratio is an experiment, not a constant

"90% synthetic, 10% real" applied blindly is a guess. Sweep it (0 / 25 / 50 / 75 / 100) and read the gold-test curve. The right ratio depends on the task, the domain and how good your generator is — and it changes when either improves.

:::warn Two honest limits of small-data utility tests
- With a few hundred rows the gold-set numbers move by several points between seeds. Run the sweep with 3 seeds and compare the spread before believing a 1-point difference.
- Utility on a downstream classifier is a proxy for "is this data good for fine-tuning an LLM". They correlate, but if your end goal is an LLM, eventually you must run the LLM experiment (Volume 2, chapter 9).
:::

:::check
- A gold test set exists, is real, and has never touched generation.
- All five models trained and compared on it.
- Mixing ratio chosen from the curve, not from a blog post.
:::
