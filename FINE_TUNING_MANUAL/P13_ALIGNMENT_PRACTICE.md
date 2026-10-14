# P13 — Step 12 (optional): DPO / ORPO / KTO / GRPO on your SFT model

**Time:** 2-4 hours. **Cost:** 1-3× an SFT run (GRPO generates G answers per prompt: the most expensive).
**GOAL:** improve **which** answers the model prefers (style, safety, correctness) on top of SFT, measured on the same eval suites, without breaking what SFT taught. Theory and measured failure modes: T05.

## Do you need this step?

| Situation | Do |
|---|---|
| SFT eval PASS and users are happy with the style | **skip it** (most projects) |
| two acceptable answers, one clearly preferred (tone, length, hedging, safety) and you can collect pairs | **DPO** |
| you have pairs from the start, little GPU memory, no SFT model yet | **ORPO** (from the base/instruct model) |
| you have thumbs-up/down logs, not pairs | **KTO** |
| answers are **checkable** (a number, a label, JSON, code with tests) | **GRPO** with a verifier |

## SECTION 1 — Prepare the SFT model as the starting point

**WHERE:** GPU terminal, `.venv-train`
**GOAL:** a merged SFT model folder (DPO/KTO/GRPO start from it; the new adapter is trained on top)

```bash
python -m peftkit.merge --adapter outputs/RUN/checkpoint-BEST --out outputs/RUN/merged
```

**YOU SHOULD SEE:** `"max_abs_logit_diff": ...` below 1e-2 in bf16 (the merge is exact; P14 Section 1).

## SECTION 2 — Build the data

| Algorithm | One line of JSONL | How to get it |
|---|---|---|
| DPO / ORPO | `{"prompt": [{"role":"user","content":"..."}], "chosen": [{"role":"assistant","content":"good"}], "rejected": [{"role":"assistant","content":"worse"}]}` | `python -m peftkit.data.preference --input data/processed/train.jsonl --out data/processed/preference.jsonl --k 4 --min-margin 2 --judge-model JUDGE` (samples K answers, judges them, keeps only clear, position-swap-consistent winners), or human comparisons |
| KTO | `{"prompt": [...], "completion": [{"role":"assistant","content":"..."}], "label": true}` | production thumbs-up (`true`) / thumbs-down (`false`) logs |
| GRPO | the **eval format**: `{"messages": [...user...], "expected": "6 bar", "must_include": ["6 bar"], "must_match": "^...$"}`, or plain SFT rows (their answer becomes `expected`) | your training prompts that have checkable answers. **Never your test set** |

> **WATCH OUT:** preference data inherits the judge's biases: position (handled by the swap), verbosity (check mean length before/after), self-preference (use a different judge model than the generator). 1k-10k clean pairs beat 100k noisy ones.

## SECTION 3 — Train (one command per algorithm)

**WHERE:** GPU terminal (tmux)
**GOAL:** a LoRA adapter in `outputs/RUN-ALGO/adapter` on top of the merged SFT model

```bash
# DPO (beta 0.1 = how far the model may drift from the SFT reference)
python -m peftkit.align dpo --model outputs/RUN/merged --data data/processed/preference.jsonl --out outputs/RUN-dpo --epochs 1

# ORPO (SFT + preference in one stage; can start from MODEL_ID directly)
python -m peftkit.align orpo --model MODEL_ID --data data/processed/preference.jsonl --out outputs/RUN-orpo --epochs 1

# KTO (unpaired good/bad labels)
python -m peftkit.align kto --model outputs/RUN/merged --data data/processed/kto.jsonl --out outputs/RUN-kto

# GRPO (verifier = the same scoring as peftkit.evaluate; G = 8 samples per prompt)
python -m peftkit.align grpo --model outputs/RUN/merged --data data/processed/rl_prompts.jsonl --out outputs/RUN-grpo --num-generations 8 --max-completion-length 256 --max-steps 300
```

| Default | DPO | ORPO | KTO | GRPO |
|---|---|---|---|---|
| lr (LoRA) | 5e-5 | 8e-5 | 5e-5 | 1e-5 |
| beta | 0.1 (KL strength) | 0.1 (λ, odds-ratio weight) | 0.1 | 0.0 (TRL 1.13 default: no KL term) |
| models in memory | policy + (adapter-disabled) reference | 1 | policy + reference | 1 (beta 0) |

**YOU SHOULD SEE (end of the run):** `{"algorithm": "dpo", "global_step": N, "adapter": "outputs/RUN-dpo/adapter", "last_log": {...}}`. Every algorithm passed this command on the toolkit's tiny model in the test suite. Real-model runs were not executed here (no GPU): do a `--max-steps 20` dry run first.

**What to watch** (`outputs/RUN-ALGO/align_meta.json` → `log_history`):

| Algorithm | Healthy | Unhealthy → action |
|---|---|---|
| DPO | `rewards/accuracies` → 1.0, `rewards/margins` grows, loss falls from 0.693 | accuracy stuck at 0.5: pairs have no real gap → raise `--min-margin`; text gets long: length bias → re-judge with a length-neutral rubric |
| ORPO | `nll_loss` falls, `log_odds_ratio` rises toward 0 | nll rising: lr too high |
| KTO | loss falls, KL bounded | collapse to very short answers: imbalanced labels → balance good/bad |
| GRPO | `reward` rises, `reward_std` > 0, `frac_reward_zero_std` < 1 | `frac_reward_zero_std` ≈ 1: every sample scores the same (too hard or too easy) → change the prompts; reward rises but answers degrade → **reward hacking**: read samples, tighten `must_match` |

## SECTION 4 — Evaluate exactly like SFT, and keep it only if it wins

```bash
python -m peftkit.evaluate --model outputs/RUN-dpo/adapter --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl safety=data/eval/safety.jsonl --max-new-tokens 256 --out reports/aligned.json
python -m peftkit.evaluate --compare reports/tuned.json reports/aligned.json
```

Keep the aligned model only if it beats the **SFT** model (`tuned.json`) on the metric you aligned for **without** dropping the domain or general suites. Then merge it onto the SFT model:

```bash
python -m peftkit.merge --adapter outputs/RUN-dpo/adapter --base outputs/RUN/merged --out outputs/RUN-dpo/merged
```

---

## CHECKPOINT

- [ ] I chose the algorithm from the table (or skipped this step with a reason).
- [ ] The data has real quality gaps (pairs) or a real verifier (GRPO), and excludes the test set.
- [ ] The aligned model beats the SFT model on its target and does not regress elsewhere.
