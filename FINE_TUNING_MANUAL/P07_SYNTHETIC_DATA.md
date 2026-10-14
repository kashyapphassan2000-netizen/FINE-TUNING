# P07 — Step 6: No data (or not enough)? Generate it, measured against reality

**Time:** 2 hours. **Cost:** $0 for the statistical route; LLM API tokens for the LLM routes.
**GOAL:** produce a synthetic dataset that **behaves like the real thing**: same distributions, same relationships between the independent variables (features) and the dependent variable (target), rare events present, no real record leaked. Then give it a name and a version, convert it to chat JSONL, and publish it.

> **Brutal honesty:** "99.9% like the real data" is not something you can claim by looking at samples. It is a **measured distance** between two distributions, driven down by a loop, plus one decisive test: a model trained on the synthetic data, **tested on real data**, must do as well as one trained on real data (TSTR). Without a real sample (even 200 rows) you cannot measure anything; you are generating plausible fiction. Get a real sample first.

## Which route? (decide in one minute)

| What you have | Route | Section |
|---|---|---|
| documents (manuals, policies, reports) | LLM writes grounded Q&A from each chunk | P06 Section 3 |
| a few dozen example tasks written by experts | **Self-Instruct**: an LLM expands 20+ seeds into hundreds of diverse tasks | Section 1 |
| a real sample of structured records (CSV/Excel/DB export: tickets, appointments, transactions, sensor logs) | **mimic reality**: profile → drivers → contract → generate → measure → gates | Sections 2-8 |
| records with text fields (ticket text, clinical notes) | hybrid: statistical generator for the columns + LLM for the text | Section 9 |
| nothing at all | interview 2-3 domain experts and write 100 rows by hand. They are your seed AND your gold test set | — |

---

## SECTION 1 — Self-Instruct: from 20+ seeds to hundreds of tasks

**WHERE:** PowerShell, `(.venv)`; `.env` has your OpenAI-compatible API
**GOAL:** `data/interim/self_instruct.jsonl` with diverse, filtered instructions and answers

Step 1. Write `data/seeds.jsonl`: 20-30 real tasks of your domain, one per line (P04 Prompt 3 drafts them; a domain expert must check them):

```json
{"instruction": "Explain why a centrifugal pump must be primed before start-up."}
{"instruction": "List the steps to replace a mechanical seal."}
{"instruction": "Classify this fault: 'pressure drops when flow rises'."}
```

Step 2. Paste and press Enter:

```powershell
python -m peftkit.data.self_instruct --seeds data/seeds.jsonl --domain "DOMAIN" --n 300 --out data/interim/self_instruct.jsonl
```

**YOU SHOULD SEE (real output on 6 seeds, `--n 12`, against the toolkit's fake LLM server):**

```text
{"proposed": 24, "length": 0, "banned_word": 6, "bad_start": 0, "too_similar": 6, "rounds": 6, "stalled": false,
 "written": 12, "usage": {"calls": 18, ...}}
```

| Counter | Meaning |
|---|---|
| `too_similar` | rejected: ROUGE-L ≥ 0.7 against an existing task (the diversity filter) |
| `banned_word` | rejected: needs images/files/audio ("draw", "plot", "image") |
| `stalled: true` | 20 rounds in a row produced nothing new: the generator is repeating itself, so it stopped paying for calls. Add more varied seeds |

Step 3. Convert to chat format and merge with your other data (P06 Section 5 Step 2): `python -m peftkit.data.format --inputs data/interim/self_instruct.jsonl ... --out-dir data/processed`.

Step 4. Read 30 rows. Self-Instruct answers are only as correct as the generator: filter with `peftkit.data.judge` and have an expert spot-check.

---

## SECTIONS 2-8 — Mimic a REAL dataset (the measured route)

Worked example used below (runs on your PC): **clinic appointment no-shows**, 1,200 rows, no personal data. Replace it with your export. The generator script plants known drivers so you can check that the profiler finds them:

```powershell
python examples/real_sample/make_clinic_sample.py
```

**YOU SHOULD SEE:** `wrote ...\clinic_noshow.jsonl (1200 rows, no-show rate 0.142)`

## SECTION 2 — Turn your export into JSONL records

**WHERE:** PowerShell
**GOAL:** one JSON object per row, with clean column names, no direct identifiers

Step 1. Remove identifiers **before** anything else: names, phone numbers, emails, national IDs, exact addresses, dates of birth, free-text notes that contain names. Keep only what predicts the target (age **band**, not birth date; region, not address).

Step 2. Convert CSV/Excel to JSONL records (one command; replace the file names):

```powershell
python -c "import json,sys; from peftkit.data.records import load_rows; rows=load_rows(sys.argv[1]); open(sys.argv[2],'w',encoding='utf-8').write(''.join(json.dumps(r, default=str)+'\n' for r in rows)); print(len(rows),'rows')" data/real/export.xlsx data/real/real.jsonl
```

> **WATCH OUT:** your real export must **never** be committed to the public repo. Keep it under `data/raw/` or `data/real/` and add `data/real/` to `.gitignore` if you use that folder.

## SECTION 3 — Profile reality: distributions, relationships, and the DRIVERS of the target

**WHERE:** PowerShell
**GOAL:** know what the generator must reproduce: every column's distribution, the correlations between features, and **which independent variables drive the dependent variable**

Step 1. Paste and press Enter (list your numeric and categorical columns; `--target` is the dependent variable):

```powershell
python -m synthkit.profile --real examples/real_sample/clinic_noshow.jsonl --numeric lead_time_days prior_no_shows distance_km --categorical age_band visit_type sms_reminder no_show --target no_show --out reports/clinic_profile.json
```

**YOU SHOULD SEE (real output):**

```text
rows: 1200
  numeric  lead_time_days           mean 13.649  std 9.597  min 0.2  median 11.5  max 59.9  skew 1.4367  outliers(3sd) 0.016
  numeric  prior_no_shows           mean 0.737  std 0.869  min 0  median 1  max 5  skew 1.2512  outliers(3sd) 0.008
  numeric  distance_km              mean 9.338  std 7.464  min 1  median 7.3  max 74.1  skew 2.7916  outliers(3sd) 0.017
  category age_band                 4 values: 45-64=0.31833, 30-44=0.2875, 18-29=0.23333, 65+=0.16083
  category visit_type               4 values: follow_up=0.4475, new_patient=0.25917, procedure=0.14833, telehealth=0.145
  category sms_reminder             2 values: yes=0.7175, no=0.2825
  category no_show                  2 values: False=0.8575, True=0.1425
  numeric correlation matrix (Pearson):
    lead_time_days            1.000  -0.028  -0.016
    prior_no_shows           -0.028   1.000  -0.011
    distance_km              -0.016  -0.011   1.000

DRIVERS of the dependent variable 'no_show' (categorical), strongest first:
  prior_no_shows           correlation_ratio  strength 0.195 (weak)  {"eta_squared": 0.0379, "feature_mean_by_class": {"False": 0.6676, "True": 1.152}}
  sms_reminder             cramers_v          strength 0.141 (weak)  {"cramers_v": 0.1413}
  age_band                 cramers_v          strength 0.117 (weak)  {"cramers_v": 0.117}
  visit_type               cramers_v          strength 0.079 (negligible)
  lead_time_days           correlation_ratio  strength 0.064 (negligible)  {"feature_mean_by_class": {"False": 13.3978, "True": 15.1602}}
  distance_km              correlation_ratio  strength 0.030 (negligible)
```

**How to read the drivers (independent → dependent):**

| Feature type → target type | Statistic | Reads as |
|---|---|---|
| numeric → numeric | Pearson r (linear) and Spearman ρ (monotonic) | −1…+1; sign = direction |
| categorical → numeric | η² (share of the target's variance explained by the category) + mean per group | 0…1 |
| numeric → categorical | correlation ratio + feature mean per class | 0…1; compare the class means |
| categorical → categorical | Cramér's V | 0 = independent, 1 = determined |

Here the profiler recovered the planted drivers **in the right order**: patients with prior no-shows miss more (mean 1.15 vs 0.67), reminders help, younger bands miss more, and lead time matters a little (15.2 vs 13.4 days). The labels say "weak" because a 14% binary outcome caps these statistics. **Read the ranking and the group means, not the adjectives.** These are associations, not causes: confirm them with a domain expert before encoding them as rules.

Step 2. Look for the four things generators most often get wrong: **skew** (lead time is right-skewed: 1.44), **rare categories** (flagged `RARE`), **outliers** (the 3σ rate), and **the target base rate** (14.25%). Write them down; Section 6 checks each one.

## SECTION 4 — Write the data contract (the rules, before generating anything)

**WHERE:** VS Code → `contracts/DATASET_NAME.yaml`
**GOAL:** the schema, ranges, categories, business constraints, privacy and quality thresholds every synthetic batch must pass

Step 1. Copy `contracts/clinic_noshow.yaml` (shipped in the repo) and edit it for your columns:

```yaml
name: clinic_noshow_synthetic
version: "1.0"
fields:
  - {name: age_band, dtype: category, categories: ["18-29", "30-44", "45-64", "65+"]}
  - {name: visit_type, dtype: category, categories: [new_patient, follow_up, procedure, telehealth]}
  - {name: lead_time_days, dtype: float, min: 0, max: 120}
  - {name: prior_no_shows, dtype: int, min: 0, max: 20}
  - {name: sms_reminder, dtype: category, categories: ["yes", "no"]}
  - {name: distance_km, dtype: float, min: 0.5, max: 150}
  - {name: no_show, dtype: bool}
constraints:                                   # Python expressions over the column names; every row must pass
  - "lead_time_days >= 0"
privacy: {pii_allowed: false, max_exact_match_rate: 0.0, min_dcr_ratio: 0.9}
quality: {max_psi: 0.2, min_utility_ratio: 0.9, max_duplicate_rate: 0.02}
```

| Field | Meaning |
|---|---|
| `dtype` | int, float, category, bool, text, datetime |
| `min` / `max` / `categories` | hard bounds: out-of-range rows are violations |
| `constraints` | business rules, e.g. `"discharge_date >= admit_date"`, `"severity <= 3 or issue_type != 'howto'"` |
| `min_dcr_ratio` | privacy: synthetic rows must be at least 0.9× as far from real rows as real rows are from each other |
| `max_exact_match_rate` | privacy: share of synthetic rows identical to a real row (0 = none allowed) |
| `max_psi` | fidelity: Population Stability Index per column (< 0.1 stable, 0.1-0.25 moderate, > 0.25 large shift) |
| `min_utility_ratio` | utility: TSTR / TRTR (Section 6) |

## SECTION 5 — Generate, measure, gate (one command)

**WHERE:** PowerShell
**GOAL:** a synthetic batch and a SHIP / SHIP WITH WARNINGS / REJECT decision with ten gates

Step 1. Paste and press Enter:

```powershell
python -m synthkit.pipeline --real examples/real_sample/clinic_noshow.jsonl --contract contracts/clinic_noshow.yaml --numeric lead_time_days prior_no_shows distance_km --categorical age_band visit_type sms_reminder --label no_show --n 3000 --smoothing 0.5 --label-mode conditional --out reports/synth_clinic
```

**YOU SHOULD SEE (real output):**

```text
[PASS] 1 schema               value=0            threshold=0 violations
[PASS] 2 constraints          value=0            threshold=0 violations
[PASS] 3 excess duplicates    value=0.0003       threshold=<= 0.02 above the real baseline
[PASS] 4 privacy: exact match value=0.0          threshold=<= 0.0
[PASS] 5 privacy: DCR ratio   value=0.967        threshold=>= 0.9
[PASS] 6 privacy: MIA AUC     value=0.5233       threshold=<= 0.6
[PASS] 7 PII                  value=0            threshold=none
[PASS] 8 distribution         value=0.1563       threshold=max PSI <= 0.2
[PASS] 9 combination coverage value=1.0          threshold=>= 0.90 of real category combinations
[PASS] 10 utility (TSTR/TRTR) value=1.1123       threshold=>= 0.9
scorecard: {'fidelity': 69, 'coverage': 100, 'diversity': 100, 'utility': 100, 'privacy': 92, 'cleanliness': 100}
decision: SHIP
utility (auc): real-only {'n_train': 720, 'accuracy': 0.8792, 'auc': 0.6313} | synth-only {'n_train': 3000, 'accuracy': 0.875, 'auc': 0.7022} | ratio 1.1123
wrote reports/synth_clinic/synthetic.jsonl and reports/synth_clinic/report.json
```

**What happens inside:**

```text
real rows -> split 60% train / 20% holdout (privacy attack) / 20% gold (utility test, NEVER used to fit)
 -> features: Gaussian COPULA (keeps each column's marginal AND the rank correlations between columns; numeric
              values mapped through each category-combination's own quantiles when the group has >= 25 rows)
 -> label (--label-mode conditional): drawn from P(target | features), a regularised logistic model fitted on the
              real train rows, SAMPLED (not argmax) so the base rate and uncertainty are kept
 -> contract enforcement (ints rounded, bounds clipped) -> 10 gates -> report
```

## SECTION 6 — Read the gates, and the loop that closes the gap

| Gate | Hard? | Fails when… | Fix |
|---|---|---|---|
| 1-2 schema / constraints | hard | a row breaks a type, range or rule | fix the contract or the generator's bounds |
| 3 duplicates | soft | more duplicates than the real data has | larger `--n` relative to distinct combinations, or more smoothing |
| 4 exact match | hard | a synthetic row IS a real row | more `--smoothing`; drop high-cardinality ID-like columns |
| 5 DCR ratio | hard | synthetic rows sit too close to real ones (memorised) | more `--smoothing` (measured below) |
| 6 membership-inference AUC | hard | an attacker can tell which records were in training | more smoothing, fewer columns, coarser bins |
| 7 PII | hard | an email/phone/card pattern appears | remove the source column, scrub text |
| 8 distribution (PSI) | soft | a column's histogram drifted | less smoothing, or fix that column |
| 9 combination coverage | soft | real category combinations are missing | more `--n`; check rare categories |
| 10 utility TSTR/TRTR | hard | a model trained on synthetic does worse **on real data** | `--label-mode conditional`; check the drivers (Section 3) are reproduced |

**The privacy ↔ fidelity loop, measured on this dataset (real runs):**

| Generator | `--smoothing` | DCR ratio (≥ 0.9) | max PSI (≤ 0.2) | utility AUC ratio (≥ 0.9) | decision |
|---|---|---|---|---|---|
| copula only (`--label-mode copula`) | 0.25 | 0.755 FAIL | 0.026 | 0.847 FAIL | REJECT |
| copula only | 0.6 | 1.02 | 0.211 WARN | 0.830 FAIL | REJECT |
| copula only | 1.0 | 1.37 | 0.480 WARN | 0.857 FAIL | REJECT |
| **conditional label** | 0.25 | 0.79 FAIL | ok | 1.077 | REJECT |
| **conditional label** | **0.5** | **0.967** | **0.156** | **1.112** | **SHIP** |

**Three lessons in that table:**

1. **Privacy and fidelity pull in opposite directions.** With `smoothing 0` every synthetic number is a real observed value (rows sit next to real rows: DCR fails). More jitter protects privacy but blurs the distributions (PSI rises). Turn the knob until both gates pass.
2. **Keeping marginals is not keeping relationships.** The copula-only generator kept every column's distribution (PSI 0.03) but **lost the drivers of the target** (utility AUC fell from 0.63 to 0.53): too many small category groups. Sampling the label from P(target | features) fixed it.
3. **Accuracy lies on imbalanced targets.** With 14% no-shows, "always says shows up" scores 86% accuracy. The utility gate therefore uses **AUC for binary targets** (an earlier version used accuracy, reported 0.995 utility for the broken generator, and has been fixed; see TOOLKIT_CHANGES.md). Utility **above 1.0** happens when a smooth label model plus more rows beats 720 noisy real rows; treat it as a reason to double-check, not as a trophy.

## SECTION 7 — Rare events, noise, counterfactuals, drift (calibrated, not random)

**WHERE:** a Python script or notebook, `(.venv)` active
**GOAL:** the cases your model must handle but the real sample barely contains

```python
import json
from synthkit import perturb
rows = [json.loads(l) for l in open("reports/synth_clinic/synthetic.jsonl")]

# measurement noise: rel_sigma must come from the REAL measurement error, not taste
noisy = perturb.feature_noise(rows, ["distance_km"], rel_sigma=0.03)
# label noise that mirrors real annotation error (makes evaluation honest)
labelled = perturb.label_noise(rows, "no_show", rate=0.02, classes=[True, False])
# missing values at the real missingness rate
gappy = perturb.missingness(rows, ["distance_km"], rate=0.05)
# counterfactuals: change ONE factor, hold everything else (fairness and robustness tests)
cf = perturb.counterfactual(rows[:200], "sms_reminder", "yes")
# distribution shift: next quarter, a new region, a policy change
shifted = perturb.drift(rows, "lead_time_days", shift=0.3)
```

**Rare events are the point.** The 0.4% that is fraud, failure or an emergency is why the system exists. `perturb.boost_rare(rows, is_rare, target_rate, generator)` raises a rare class to a target share; **record the original base rate** and re-weight at training/evaluation time rather than pretending the world changed (labs 42 and 41 measure this and model collapse).

## SECTION 8 — Name it, version it, convert it to chat JSONL, publish it

**WHERE:** PowerShell
**GOAL:** `DATASET_NAME-v1.0/` with `train.jsonl`, `eval.jsonl`, a dataset card and lineage on every row, optionally on your Hub account

Step 1. Records → chat conversations (here: predict the target from the features; `--label-map` turns raw values into answers):

```powershell
python -m peftkit.data.records --input reports/synth_clinic/synthetic.jsonl --name clinic-noshow --version 1.0 --task predict --target no_show --features age_band visit_type lead_time_days prior_no_shows sms_reminder distance_km --system "You are a clinic operations assistant. Predict whether an appointment will be missed." --label-map True="Likely to miss the appointment." False="Likely to attend." --out data/datasets
```

**YOU SHOULD SEE (real output):**

```text
{ "name": "clinic-noshow", "version": "1.0", "task": "predict", "rows": 3000, "conversations": 3000,
  "train": 2700, "eval": 300, "dir": "data/datasets/clinic-noshow-v1.0" }
```

One row of `data/datasets/clinic-noshow-v1.0/train.jsonl` (real):

```text
user: Given this record:
- age_band: 45-64
- visit_type: follow_up
- lead_time_days: 21.94
- prior_no_shows: 0
- sms_reminder: yes
- distance_km: 16.21
What is the value of no_show?
assistant: Likely to attend.
```

| `--task` | Use it for |
|---|---|
| `predict` | the model answers the target from the features (3 built-in phrasings, picked per row) |
| `describe` | the model writes a description of a record |
| `template` | **your** wording: `--user "A {age_band} patient booked a {visit_type} {lead_time_days} days ahead. Risk?" --assistant "..."`; repeat `--user` 3-5 times for phrasing diversity |

Step 2. Publish the dataset to your Hub account (private first):

```powershell
python -m peftkit.publish --src data/datasets/clinic-noshow-v1.0 --hub HF_USER/clinic-noshow --repo-type dataset --private
```

**YOU SHOULD SEE:** `hub: https://huggingface.co/datasets/HF_USER/clinic-noshow/...`. The folder's `README.md` becomes the dataset card.

> **Brutal honesty about this example:** for **pure tabular prediction** (a yes/no from 6 columns), a gradient-boosted tree (XGBoost/LightGBM) is cheaper, faster and usually **more accurate** than a fine-tuned LLM. Fine-tune an LLM on records when the output must be **language**: an explanation for the scheduler, a message to the patient, a triage note, or when the records come with free text. Use the synthetic records for the classical model and the LLM together if needed.

## SECTION 9 — Records with text fields (hybrid generation)

Structured columns from the copula (numbers statistically right), text from a template or an LLM conditioned on those columns (language human):

```python
from synthkit.generate import hybrid
rows = hybrid(rows, "note", lambda r: f"{r['visit_type'].replace('_', ' ')} booked {r['lead_time_days']:.0f} days ahead; "
                                      f"{'reminder sent' if r['sms_reminder'] == 'yes' else 'no reminder'}.")
```

For LLM-written records constrained by your schema and real examples: `synthkit.generate.llm_records(llm, domain, schema, n, constraints, seed_examples)` (lab 44), then judge them with `synthkit.judge.multi_judge` (several judges, disagreement flagged for a human) and **calibrate the judge against 50 human labels** (`synthkit.judge.best_threshold`; Cohen's κ < 0.4 means the judge is not measuring what your humans measure).

## Where to get a real anchor sample (free sources)

```powershell
python -m synthkit.vendors --domain healthcare --free-only
```

**YOU SHOULD SEE (real output, abridged):**

```text
source                            domain                auth                              cost      commercial use
OpenFDA                           healthcare/drugs      free key optional                 free      allowed
PhysioNet / MIMIC                 healthcare/clinical   credentialed (training + DUA)     free      strict DUA; no redistribution
limits move; verify on the provider page. free tier != commercial licence.
```

---

## CHECKPOINT

- [ ] I have a real sample (≥ 200 rows) with identifiers removed, as JSONL.
- [ ] I profiled it and wrote down the target's drivers, the skewed columns, the rare categories and the base rate.
- [ ] My contract YAML exists, and the pipeline says SHIP (or SHIP WITH WARNINGS that I understand).
- [ ] Utility was judged with the right metric (AUC for an imbalanced binary target).
- [ ] The dataset has a name and version, chat JSONL, a card, lineage, and is published privately.
