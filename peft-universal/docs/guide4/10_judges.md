# 10. Step 7: judging — never trust a single judge

@file src/synthkit/judge.py

## 10.1 Decomposed criteria, not a 1-10 score

A single number hides why a record is bad. Score each criterion 0-4 and weight them by the **business objective**:

$$ score = (0.35·correctness + 0.20·completeness + 0.20·instruction + 0.15·safety + 0.10·style) / 4

Change the weights when the business changes: a compliance dataset weights safety highest; a style-transfer dataset weights style. Arbitrary weights are the most common silent error here.

## 10.2 Several judges, and what disagreement means

`multi_judge` runs N judges and returns mean, median, spread and `needs_human`. Spread is the signal: when judges disagree, a human must look. Known judge biases to defend against: verbosity (longer looks better), position (A vs B ordering), self-preference (a model prefers its own style), and leniency drift over a long run. Defences: several judges, criteria-specific prompts, position randomisation, temperature 0, and a judge model different from the generator.

## 10.3 Calibrate against humans before you trust any of it

`agreement()` and `best_threshold()` compare judge scores with human keep/reject labels and report accuracy and **Cohen's kappa**. Kappa below ~0.4 means your judge is not measuring what your humans measure — fix the rubric, not the threshold.

@lab lab44_llm_records_judges

In that run the judges correctly rank the two genuine tickets above a templated one and one containing personal data, the weakest criterion is named per record (style for one, correctness for the template, safety for the PII case), the threshold search recovers perfect agreement with the human labels, and the PII scan flags the email and phone number that the judge alone would not have blocked.

## 10.4 Human review that costs an hour a week

Sample 50 records per batch, stratified: 20 highest-scoring, 20 lowest, 10 where judges disagreed most. Label keep/reject. Feed those labels back into `agreement()`. That is your calibration set, your judge regression test, and the basis of your audit trail. Argilla (Volume 1, chapter 18) is the tool for the UI part.

:::check
- Weights come from the business objective, written down.
- At least two judges, position randomised, generator ≠ judge.
- Kappa measured against ≥50 human labels before the judge gates anything.
:::
