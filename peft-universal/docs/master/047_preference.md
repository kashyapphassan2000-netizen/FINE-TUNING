# 47. Preference dataset generation

:::why In one line
SFT teaches what a good answer looks like. Preference data (chosen vs rejected) teaches which of two answers is better, used by DPO/ORPO/KTO after SFT.
:::

## 47.1 Recipe used here

1. Take prompts (the SFT prompts, or new ones).
2. Sample K candidates: the same model at different temperatures, optionally a weaker model (`--weak-model`) to guarantee some bad answers.
3. Score every candidate with a pointwise judge (1-5).
4. chosen = best, rejected = worst; keep only if the score gap ≥ `min_margin` (small gaps are noise).
5. Confirm with a pairwise judgment in both orders; drop if the judge is inconsistent.
6. Remove identical candidates (no signal).

This is the same idea as UltraFeedback (Cui et al., 2023), which rated completions from several models with GPT-4 on several aspects.

@file src/peftkit/data/preference.py

```any
python -m peftkit.data.preference --input data/processed/train.jsonl --out data/processed/preference.jsonl \
   --k 4 --min-margin 2 --judge-model <different-model> --limit 50
```

The test `test_judge_and_preference` checks that the GOOD candidate becomes chosen, the BAD one rejected, and the margin rule holds.

:::warn Scope
This guide builds the preference file. The DPO training step itself (TRL `DPOTrainer` on top of your SFT adapter) is not part of this template's tested code. The output format is TRL's conversational preference format, so it plugs in directly.
:::

:::check
- Each pair has a real quality difference when I read 20 of them.
- The chosen answer is not systematically just the longer one (verbosity bias, chapter 48).
:::
