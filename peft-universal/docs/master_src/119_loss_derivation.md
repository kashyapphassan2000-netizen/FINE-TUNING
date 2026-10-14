# 119. Universal loss derivation

:::why In one line
The objective is implied by the **contracts**, not by the model's name: what goes in, what comes out, and what those numbers mean.
:::

@file src/uadapt/derive.py

## 119.1 The rules, in order, each with its rationale

|#|Evidence|Derived objective|Confidence|Wrong if|
|---|---|---|---|---|
|1|the model returns `loss` for a batch containing labels|the model's own head loss|high|you need a different objective than the head implements|
|2|`forward()` takes a `timestep` and returns a tensor shaped like its input|diffusion ε-MSE|medium|the model predicts v or x₀ instead of ε|
|3|paired inputs + **scalar** output|Bradley-Terry pairwise (reward model)|high|you wanted DPO (that trains a policy, not a scorer)|
|4|paired inputs + embedding output|multiple-negatives contrastive|high|the positives are not truly related, or the batch is too small|
|5|3-D logits, width = vocabulary|token cross-entropy (shifted for causal LMs)|high|labels are not aligned the way the head expects|
|6|2-D logits, width = num_labels|cross-entropy over classes|high|the task is multi-label (use BCE)|
|7|3-D logits, width = num_labels|per-token cross-entropy|high|sub-word alignment is wrong|
|8|**width 1**|MSE / Huber (value), or BCE if it is a binary logit|medium|it is a reward score (then pairwise)|
|9|output is a hidden state / embedding, no pairs|**undetermined**|low|— supply pairs, or add a head|
|10|anything else|**CUSTOM ENGINEERING REQUIRED**|none|— write `loss_fn(model, batch)`|

Rule 8 is checked **before** rule 6: a "1-class classifier" is a value, never a class distribution. That ordering bug was found by the tests, not by reading.

## 119.2 Using it

```py
from uadapt.derive import derive_loss, input_contract, output_contract
print(output_contract(model, batch))          # what the numbers mean
d = derive_loss(model, batch, pairs=False)
print(d["name"], d["confidence"], d["why"], d["wrong_if"])
loss_fn = d["loss_fn"]                        # None means: you must write it
```

## 119.3 Why "I don't know" is a feature

A framework that always returns *some* loss will happily train your model on the wrong objective, and the loss curve will look fine. This one refuses, names the evidence it had, and tells you what to write. Chapter 121's invented model hits exactly that branch.

:::check
- I read the derived loss's `why` and `wrong_if` before training.
- Where confidence was medium or low, I confirmed against the model's own training code.
:::
