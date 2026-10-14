# 5. Domain-specific fine-tuning: best practices

:::why In one line
A domain model is judged on entities, phrasings and edge cases it has never seen. Coverage of your domain beats volume of your domain, and you must keep checking what it forgot.
:::

## 5.1 The measurements

@file labs/lab54_domain_sft.py
@lab lab54_domain_sft

Two experiments. **A** scales the training data and tests on machines **and** a phrasing never seen in training. **B** teaches two capabilities, then fine-tunes again on the domain with and without a small replay of the other capability, and measures both.

**Section A**, honestly: 4, 16 and 32 rows all sit at chance (0.25) on the unseen phrasing. Only the widest configuration — 6 machines × 4 templates, 96 rows — reaches 0.75. Note what changed: not just the row count, but the **variety** of entities and phrasings, exactly as lab 51 predicted. The absolute numbers belong to a 3-layer random model; the shape of the result is the lesson.

**Section B** is the trade-off, measured. Stage 0 teaches two capabilities (general accuracy 1.00). Then, continuing from that merged model:

|stage 1 training data|domain accuracy|general accuracy|
|---|---|---|
|domain only|0.75|**0.67** (forgot part of the other capability)|
|domain + 20% replay|0.625|**1.00** (kept it)|

Replay preserved the other capability and cost a little domain accuracy. That is the real decision you face, not a free lunch: pick the trade deliberately and measure both sides. If you train a fresh adapter on the base model instead of continuing from the deployed one, you are not measuring forgetting at all — an earlier version of this lab made exactly that mistake.

## 5.2 The practices that transfer to real models

|#|Practice|Why|
|---|---|---|
|1|hold out **entities**, not just rows|testing on rows about machines you trained on measures memorisation|
|2|cover every intent × every phrasing × every entity type|your users will not use your template|
|3|mix 5-30% general or previous-capability data|prevents forgetting (measured in section B and in Volume 2, chapter 10)|
|4|vary the system prompt; include the empty one|otherwise behaviour is tied to one string|
|5|include refusals, "not in the documents", and clarifying questions|otherwise the model invents|
|6|1-3 epochs, lr 1e-4..3e-4, r=16 on a real model|Volume 1, chapter 9; this toy needs far more because it is random-init|
|7|watch eval every 20-50 steps; keep the best checkpoint|small domain sets overfit within one epoch|
|8|evaluate per segment, not just on average|the average hides your biggest customer's segment|
|9|re-run a general suite every time|domain gains that cost general ability are usually a bad trade|
|10|put the facts in retrieval, the behaviour in SFT|facts change; behaviour does not|

## 5.3 How much data do you actually need?

There is no universal number, and anyone quoting one without your task is guessing. The honest procedure:

1. Build the eval set first (50-200 real items).
2. Train on 25%, 50%, 100% of what you have.
3. Plot eval score against data size.
4. If the curve is still rising, collect more of the same. If it is flat, more of the same will not help — you need more **variety**, better labels, or a different method.

That curve costs three short runs and replaces every rule of thumb.

## 5.4 The domain SFT failure modes

|Symptom|Usual cause|
|---|---|
|great on your test prompts, useless in production|test prompts came from the same template as training|
|forgot general ability|no replay data, too many epochs, lr too high|
|invents policy details|no refusals in training; facts should be retrieved|
|inconsistent formatting|inconsistent formatting in the training answers|
|works for one customer only|entity leakage: one customer dominates the data|
|degrades after every retrain|no fixed gold set; you are chasing noise|

:::check
- My eval holds out entities and phrasings, not only rows.
- Replay data included and general suite re-run.
- Data-size curve plotted rather than a number guessed.
:::
