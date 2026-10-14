# P12 — Step 11: Hyperparameter tuning that ENDS (and maps to your benchmarks)

**Time:** 2-4 hours of GPU time for a real search on an 8B model. **Cost:** 3-5× one training run.
**GOAL:** the best configuration by **held-out score**, with the gain proven larger than seed noise, and a written stopping rule so the search ends.

> **Brutal honesty:** "play with the hyperparameters until the error reaches the global minimum" is not achievable and not even desirable. A neural network's loss surface has no findable global minimum, and training loss near zero means **memorisation**, which scores worse on new questions. What IS achievable: search a small space, kill bad runs early, keep the best by **eval score**, confirm it against seed noise, and stop when gains are smaller than that noise.

---

## SECTION 1 — What to tune, in which order (cheapest first)

Never tune two things at once. Record every run (the toolkit writes `run_meta.json` per run).

| Order | Knob | Range to try | Why it matters |
|---|---|---|---|
| 0 | **data quality** | fix the regressions from P11 | cheapest, biggest effect; no knob compensates for wrong data |
| 1 | **learning rate** | LoRA/QLoRA 5e-5 · 1e-4 · 2e-4 · 5e-4; DoRA ×0.5; full FT 5e-6 · 1e-5 · 2e-5 | the single most important knob (lab 15: 2e-5 barely moves LoRA, 5e-2 diverges) |
| 2 | `max_seq_length` | the p95 of your data | too short cuts answers; too long wastes memory |
| 3 | effective batch (`per_device × grad_accum × GPUs`) | 8 · 16 · 32 · 64 | stability vs steps per epoch; change lr with it (×√ratio) |
| 4 | **epochs** | 1 · 2 · 3 | watch eval; the best checkpoint is often mid-run |
| 5 | **rank r** (and α = 2r) | 8 · 16 · 32 (64+ for CPT) | capacity; lab 15: r 2 → 32 helped little on a small task |
| 6 | target modules | all-linear vs attention-only | MoE: attention-only; small data: attention-only can generalise better |
| 7 | dropout / weight decay | 0 · 0.05 · 0.1 / 0 · 0.01 | small datasets over-fit |
| 8 | method | LoRA → DoRA / LoRA+ / AdaLoRA (T03 §7) | only after 1-7 are settled |
| 9 | warmup / scheduler | 3% warmup, cosine | rarely the problem |

## SECTION 2 — Learning-rate and rank sweep by hand (understand before automating)

**WHERE:** GPU terminal (tmux)
**GOAL:** see the effect of one knob with everything else fixed

```bash
for LR in 5e-5 1e-4 2e-4 5e-4; do
  python -m peftkit.train --config configs/base.yaml configs/methods/lora.yaml configs/models/SLUG.yaml \
    --set train.lr=$LR train.max_steps=150 train.output_dir=outputs/sweep_lr_$LR train.save_steps=100000
done
grep -H '"eval_loss"' outputs/sweep_lr_*/run_meta.json
```

**What it looks like** (real output of `python labs/lab15_hyperparameter_sweep.py` on the toy model; your numbers differ, the shape does not):

```text
| lr      | train_loss | eval_loss | eval_token_acc |
| 2e-05   | 5.9776     | 5.9709    | 0.0074         |   <- too low: barely moves (a full-FT lr on LoRA)
| 0.0002  | 5.8551     | 5.7828    | 0.1407         |
| 0.002   | 5.4933     | 5.2617    | 0.5556         |   <- best here (a random toy model needs a high lr)
| 0.05    | 5.712      | 5.6541    | 0.0963         |   <- too high: worse again
```

## SECTION 3 — Automated search: random search + successive halving

**WHERE:** GPU terminal (tmux)
**GOAL:** many candidates, short runs, the weak half killed at each rung, budget spent on survivors

Step 1. Paste and press Enter:

```bash
python -m peftkit.hpo --config configs/base.yaml configs/methods/lora.yaml configs/models/SLUG.yaml --space '{"train.lr": [5e-5, 1e-4, 2e-4, 5e-4], "method.r": [8, 16, 32], "train.epochs": [1, 2, 3]}' --budget 12 --rungs 20 60 180 --out outputs/hpo --seeds
```

| Flag | Meaning |
|---|---|
| `--space` | JSON: each key is any config field (`train.lr`, `method.r`, `method.alpha`, `train.epochs`, `method.dropout`, …); each value is a **list of choices** |
| `--budget 12` | 12 random candidates (random search beats grid search at equal budget when only a few knobs matter) |
| `--rungs 20 60 180` | every candidate runs 20 steps; the best half continues to 60; the best half of those to 180 |
| `--seeds` | re-run the winner with seeds 0, 1, 2 to measure noise |

**YOU SHOULD SEE (real output shape, from lab 64 on the toy model):**

```text
  id     final score  rungs survived  params
  c0          5.2328               2  {'train.lr': 0.008, 'method.r': 16}
  c2          5.5783               2  {'train.lr': 0.002, 'method.r': 16}
  c3          5.6442               2  {'train.lr': 0.002, 'method.r': 4}
  c4          5.9423               1  {'train.lr': 0.0005, 'method.r': 16}
  seed noise for that config: [5.00287, 5.02587, 5.08201] spread 0.07914
  treat improvements smaller than the spread as noise, not progress
```

`score` = eval loss (lower is better). Progress is saved after every rung in `outputs/hpo/hpo.json`.

## SECTION 4 — The stopping rule (the answer to "when am I done?")

Stop when **all three** hold:

1. the best eval score stopped improving by more than the **seed spread** (0.079 in the example: a config "better" by 0.03 is noise);
2. the **general / safety** suites and the benchmark gate (P11 Section 4) are stable;
3. failure analysis (P11 Section 2) says the remaining errors are **data or task-definition** problems, not tuning ones.

Then the next gain comes from **data** (new intents, entities, phrasings, hard cases), not from knobs.

## SECTION 5 — Mapping the search to YOUR benchmark (not just eval loss)

Eval loss is a **proxy**. What you ship is the domain score, and what you must protect is the general score. The search picks candidates by eval loss (cheap, available every few steps); then **confirm the top 3 on the real objective**:

```bash
for C in c0 c2 c3; do
  D=$(ls -d outputs/hpo/${C}_r* | tail -1)/adapter
  python -m peftkit.evaluate --model $D --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl --max-new-tokens 256 --out reports/hpo_$C.json
  python -m peftkit.evaluate --compare reports/base.json reports/hpo_$C.json | tail -1
done
```

**Choose by this rule (write it down before looking at results):**

```text
maximise   domain score
subject to general score drop <= 0.02 (and benchmark gate PASS),  safety score drop <= 0
tie-break  lower rank, fewer epochs (cheaper, forgets less)
```

**A learning curve tells you whether more DATA will help** (run with 25% / 50% / 100% of the training rows, same config):

```bash
for P in 25 50 100; do
  head -n $(( $(wc -l < data/processed/train.jsonl) * P / 100 )) data/processed/train.jsonl > data/processed/train_$P.jsonl
  python -m peftkit.train --config configs/base.yaml configs/methods/lora.yaml configs/models/SLUG.yaml --set data.train_file=data/processed/train_$P.jsonl train.output_dir=outputs/curve_$P
  python -m peftkit.evaluate --model outputs/curve_$P/adapter --suite domain=data/eval/test.jsonl --out reports/curve_$P.json
done
```

If 50% → 100% gains less than the seed spread, more of the **same** data will not help (P05 §3).

---

## CHECKPOINT

- [ ] I tuned lr first, one knob at a time, and recorded every run.
- [ ] The HPO winner's gain over the runner-up is larger than the seed spread (or I picked the simpler one).
- [ ] The chosen config maximises the domain score with general drop ≤ 2 points and the benchmark gate PASS.
- [ ] I know whether more data would help (learning curve).
