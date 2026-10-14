# 16. Model collapse and the synthetic:real ratio

:::why In one line
Train on your own generator's output, generation after generation, and the distribution degrades — tails first.
:::

@lab lab41_collapse_and_mixing

## 16.1 What the measurement shows

Five generations of "fit the generator on the previous generation's output":

- **Without real data**: PSI against the original real distribution rises from 0.085 to about 0.18 and wanders; downstream accuracy on the real gold set is unstable across generations.
- **With 20% real data mixed in at every generation**: PSI stays around 0.02-0.05 and accuracy stays stable.

Honest reading of this particular experiment: the variance did **not** collapse to zero, because this generator resamples observed values rather than fitting a parametric model — the failure mode here is distribution *drift* and instability rather than dramatic variance shrinkage. Published work on recursive training with parametric models (a Nature study on recursively generated data) shows the stronger tail-loss effect. Either way the defence is the same, and it is cheap.

## 16.2 The rules

1. Keep real data in every generation. 10-30% is the usual working range; measure it.
2. Never train generation N on generation N−1 without re-anchoring to real data.
3. Keep lineage on every record (`source`, `generator_version`, `parent_batch`) so you can always reconstruct how much of your corpus is second-hand.
4. Track the synthetic share of your training corpus as a first-class metric.
5. Refresh the real seed on a schedule. A one-off seed ages.

:::check
- Synthetic share per dataset version is recorded.
- Every batch has lineage to a real source.
- The recursive experiment has been run once on my own data so I know my own degradation curve.
:::
