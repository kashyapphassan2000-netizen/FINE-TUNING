# P11 — Step 10: Evaluate — prove it improved, find what got worse, check nothing was forgotten

**Time:** 30-120 minutes. **Cost:** GPU time for generation.
**GOAL:** a before/after table on unseen questions with a PASS verdict, the list of regressions, the best checkpoint identified, and (before release) public benchmarks showing no forgetting.

> **Brutal honesty:** "the loss went down" is not a result. Training loss measures how well the model fits the training data, which is not what you sell. The only evidence is the **same held-out test, before and after**, plus a **general** suite that shows you did not break the model.

---

## SECTION 1 — Score the fine-tuned model on the same suites as the baseline

**WHERE:** GPU terminal (tmux), `.venv-train` active, `.env` loaded
**GOAL:** `reports/tuned.json`, directly comparable with `reports/base.json` (P10 Section 4)

Step 1. Paste and press Enter (the adapter folder: the base model is read from its `adapter_config.json`):

```bash
python -m peftkit.evaluate --model outputs/RUN/adapter --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl safety=data/eval/safety.jsonl --max-new-tokens 256 --out reports/tuned.json
```

(Full fine-tuning: `--model outputs/RUN/model`.)

Step 2. Compare:

```bash
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

**YOU SHOULD SEE (format; real numbers from the CPU rehearsal shown):**

```text
|suite|before|after|delta|
|---|---|---|---|
|domain|0.0|0.5|+0.500|

verdict: PASS: domain improved, no large general regression  regressions: 0
```

| Verdict | Means | Do this |
|---|---|---|
| `PASS` | domain up, general/safety within 0.05 | go on (Section 2) |
| `WARNING: … catastrophic forgetting` | domain up, general/safety down > 0.05 | add 5-30% replay data, fewer epochs, lower lr or rank; retrain (P12) |
| `FAIL: the domain did not improve` | no gain | P18 rows 5-9; check data quality and the mask before tuning anything |

**How each item is scored** (`peftkit.evaluate`): exact match after normalisation, "contains the gold answer", every `must_include` phrase present, `must_match` regex; the item score is the mean of the checks that apply. It also computes the **answer NLL** (how likely the model finds the gold answer), useful when generation is noisy.

## SECTION 2 — Read the regressions (the items that got WORSE)

**WHERE:** VS Code → `reports/compare.json`
**GOAL:** know what you broke, not just the average

Step 1. Open `reports/compare.json` → `regressions`: each entry has the suite, item id, score before/after and the new answer.

Step 2. For every regression, write one line: data problem (wrong/missing training example), task-definition problem (the test expects something else), or model problem (forgetting, format drift). Fix data problems first: they are the cheapest.

Step 3. **Per segment:** if your test items have segments (product line, customer type, language), group the per-item scores by segment. An average hides the segment your biggest customer cares about.

## SECTION 3 — Pick the best checkpoint (not the last one)

**WHERE:** GPU terminal
**GOAL:** the checkpoint with the best **eval**, which is often not the final step

Step 1. Quick view of eval loss per checkpoint (from the trainer's own log):

```bash
python -c "import json;h=json.load(open('outputs/RUN/run_meta.json'))['log_history'];[print(x['step'], round(x['eval_loss'],4)) for x in h if 'eval_loss' in x]"
```

Step 2. The decisive view: score each checkpoint on the domain suite:

```bash
for c in outputs/RUN/checkpoint-*; do python -m peftkit.evaluate --model $c --suite domain=data/eval/test.jsonl --max-new-tokens 256 --out reports/$(basename $c).json; done
grep -H '"score"' reports/checkpoint-*.json | head
```

**YOU SHOULD SEE:** one score per checkpoint. Keep the best (ties: the **earlier** one, which has forgotten less). Re-run Section 1 Step 2 with that checkpoint against the baseline.

> **WATCH OUT:** `eval_loss` rising while `train_loss` keeps falling = **overfitting**. The best checkpoint is before the rise. Lab 16 (`python labs/lab16_overfitting.py`) shows the curve.

## SECTION 4 — Public benchmarks: prove nothing general was forgotten

**WHERE:** GPU terminal, a third environment (`lm-eval` must not touch the training or serving envs)
**GOAL:** base vs tuned on MMLU / GSM8K / etc., and a forgetting gate

Step 1. Install lm-evaluation-harness in its own environment:

```bash
uv venv .venv-eval --python 3.11 && source .venv-eval/bin/activate
uv pip install "lm-eval[vllm]==0.4.13"
lm_eval --help | head -3
```

Step 2. Benchmark the BASE model (use `--limit 200` for a first pass, and the SAME limit for both runs):

```bash
lm_eval --model vllm --model_args "pretrained=MODEL_ID,dtype=bfloat16,gpu_memory_utilization=0.85" --tasks mmlu,gsm8k,arc_challenge,hellaswag --batch_size auto --apply_chat_template --limit 200 --output_path reports/benchmarks/base
```

Step 3. Benchmark the TUNED model (the adapter directly, through the `hf` backend's `peft=` argument):

```bash
lm_eval --model hf --model_args "pretrained=MODEL_ID,peft=outputs/RUN/checkpoint-BEST,dtype=bfloat16" --tasks mmlu,gsm8k,arc_challenge,hellaswag --batch_size auto --apply_chat_template --limit 200 --output_path reports/benchmarks/tuned
```

Step 4. The forgetting gate (back in the training env):

```bash
source .venv-train/bin/activate
python -m peftkit.bench_gate --base reports/benchmarks/base --tuned reports/benchmarks/tuned --max-drop 0.02 --out reports/bench_gate.json
```

**YOU SHOULD SEE (format, from the toolkit's test fixtures in lm-eval's exact result layout):**

```text
task                  metric                            base   tuned    delta
mmlu                  acc,none                          0.62   0.615  -0.0050
gsm8k                 exact_match,strict-match          0.55    0.49  -0.0600  <-- DROP
hellaswag             acc_norm,none                     0.79   0.795  +0.0050
verdict: FAIL
```

| Benchmark | Measures | Note |
|---|---|---|
| `mmlu` | broad knowledge (57 subjects), multiple choice | loglikelihood: needs the `vllm` or `hf` backend, not a chat API |
| `gsm8k` | grade-school maths, generated answers | the first thing narrow fine-tunes break |
| `arc_challenge`, `hellaswag`, `winogrande` | reasoning / commonsense | multiple choice |
| `ifeval` | instruction following (format constraints) | important after format-heavy fine-tunes |
| domain benchmarks | e.g. `medqa`/`pubmedqa` (health), `legalbench` subsets (legal), `finqa`-style (finance) | check the task names in `lm_eval --tasks list` for your installed version (verify) |

> **WATCH OUT:** compare only runs with identical `--tasks`, `--limit`, `--num_fewshot` and `--apply_chat_template`. Open both `results_*.json` and compare their `config` sections before believing any delta. lm-eval was **not executed** while writing this manual (it needs the Hub datasets and a GPU); flags were read from the lm-eval 0.4.13 source by the toolkit's authors, and the gate is tested on files in that format.

## SECTION 5 — Open-ended answers: an LLM judge (optional, with calibration)

For answers that exact match cannot score (explanations, drafts), score both models' answers with a judge and position-swapped pairwise comparison (`peftkit.data.judge.pairwise`: a verdict only counts if it survives swapping A and B). **Calibrate the judge first:** have a human label 50 items keep/reject and check Cohen's κ with `synthkit.judge.best_threshold`. κ < 0.4 = the judge does not measure what your humans measure.

---

## CHECKPOINT

- [ ] `reports/compare.json` verdict PASS, regressions read and explained.
- [ ] Best checkpoint chosen by eval score (not the last step).
- [ ] (Before release) `reports/bench_gate.json` PASS: no general benchmark dropped more than 2 points.
