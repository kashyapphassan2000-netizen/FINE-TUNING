# 11. Overfitting and underfitting

## 11.1 The two curves

```text
train loss ↓   eval loss ↓      healthy: keep going
train loss ↓   eval loss flat   diminishing returns: consider stopping
train loss ↓   eval loss ↑      OVERFITTING: stop at the best eval point
train loss flat, eval flat, both high   UNDERFITTING: lr too low, rank too small, too few steps
```

## 11.2 Measured

@file labs/lab16_overfitting.py
@lab lab16_overfitting

Eight training examples, four unseen eval examples, 60 epochs. Training loss keeps falling; eval loss reaches its best point and then climbs while eval token accuracy stalls. The lesson is the stopping rule: **the checkpoint you want is the one with the best eval score, not the last one**.

## 11.3 Fixes

|Symptom|Fixes, in the order to try them|
|---|---|
|Overfitting|fewer epochs / early stopping; more and more diverse data; lower rank; add dropout; add weight decay; check for duplicates|
|Underfitting|raise lr; more steps/epochs; raise rank; adapt more modules (`all-linear`); check the loss mask is not empty; check data actually contains the behaviour|
|Both losses fall but answers are bad|your eval measures the wrong thing; or data is fluent but wrong (ch 6)|
|Eval loss down, task score down|loss is not your metric: score the task (ch 9)|

```yaml Early stopping in this template
train:
  eval_steps: 50
  save_steps: 50
  save_total_limit: 3
# then pick the checkpoint with the best eval score and merge THAT one:
# python -m peftkit.merge --adapter outputs/<run>/checkpoint-<best> --out outputs/<run>/merged
```

:::warn Small data overfits fast
With a few hundred examples, 3 epochs can already be too many. Watch eval every 20-50 steps rather than once per epoch.
:::

:::check
- I have an eval measured during training, not only at the end.
- I know which checkpoint had the best eval score.
- I merged that checkpoint, not automatically the last one.
:::
