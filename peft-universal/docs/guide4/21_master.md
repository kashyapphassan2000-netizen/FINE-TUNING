# 21. The master loop, the capstone and the checklist

## 21.1 The 26 steps, in order

```text
 1 define the target capability        14 deduplicate
 2 define the failure modes            15 privacy scan (ch 9)
 3 define the taxonomy                 16 distribution test (ch 8)
 4 collect a small real seed (ch 5)    17 judge (ch 10)
 5 profile the real distribution (ch 4)18 human sample audit
 6 write the data contract (ch 3)      19 train
 7 build generation programs (ch 6)    20 evaluate on the REAL holdout (ch 11)
 8 generate candidates                 21 error analysis (ch 18)
 9 generate negatives                  22 find the missing regions
10 generate edge cases (ch 17)         23 generate TARGETED data
11 generate counterfactuals (ch 7)     24 retrain
12 add calibrated noise (ch 7)         25 re-evaluate
13 validate schema (ch 3)              26 version everything (ch 20)
                        └── loop, with the monitor running (ch 15) ──┘
```

## 21.2 Capstone: build one for your own business

1. Pick a decision your business makes repeatedly (approve/decline, route, price, alert, reply).
2. Write the seven-row domain mapping (chapter 19) and the contract (chapter 3).
3. Get 300+ real rows and hand-write a 50-row gold test set. Never use it for generation.
4. Profile, generate 10× your seed, perturb, validate, privacy-test, judge, gate.
5. Train the five models of chapter 11 and plot the mixing curve on your gold set.
6. Do one error-analysis cycle and one targeted regeneration (chapter 18). Measure whether it helped.
7. Set up the drift monitor with **calibrated** thresholds (chapter 15).
8. Write one page: what you built, the utility ratio, the privacy numbers, the gate decisions, and what you would not trust.

**Grading, be strict**: 0-3 = you generated data and it looked nice. 4-5 = you measured fidelity. 6-7 = privacy tested and gates enforced. 8-9 = utility proven on a real gold set and one targeted regeneration cycle measured. 10 = someone else reproduced your dataset from your contract, seed snapshot, generator config and report — and your monitor caught a real change before your metrics did.

## 21.3 One-page checklist

```text BEFORE GENERATING
  contract written (schema, constraints, privacy, quality thresholds)
  real seed obtained; licence + commercial-use checked
  gold test set built FIRST, from real data, never used again for generation
  profile of the REAL TRAIN SPLIT only

WHILE GENERATING
  generator matched to data type (copula / simulator / LLM / hybrid)
  smoothing or another privacy control ON if you resample real values
  rare events, counterfactuals and calibrated noise generated deliberately
  provenance on every record (generator, prompt, contract version, batch)

BEFORE SHIPPING (one hard failure = reject)
  schema + constraints           : 0 violations
  privacy                        : exact match ~0, DCR ratio healthy, MIA AUC ~0.5, no PII
  fidelity                       : PSI / KS / TVD / correlation delta / MMD within contract
  coverage                       : tails and category combinations present
  utility                        : TSTR/TRTR above threshold on the REAL gold set
  judge                          : calibrated against >= 50 human labels (kappa > 0.4)

AFTER SHIPPING
  drift monitor with thresholds calibrated on quiet history
  L_distribution tracked per release
  monthly utility re-check on a fresh real slice
  error analysis -> targeted regeneration -> re-measure
  version: contract + seed + generator config + gold set, together
```

## 21.4 The mental model

```text
REALITY -> MEASURE -> MODEL -> GENERATE -> PERTURB -> FILTER -> JUDGE -> TRAIN
   ^                                                                      |
   └──── GENERATE TARGETED DATA <- FIND FAILURE <- TEST ON REALITY <───────┘
```

Synthetic data generation is data engineering + statistics + generative modelling + evaluation + domain knowledge + governance. The closed loop is the product. Everything else is a demo.

## 21.5 Verification of this book

```any
python -m pytest -q tests/test_synthkit.py     # 13 tests for the toolkit
python -m pytest -q                            # the whole repository
python labs/lab40_end_to_end.py                # ... through lab44_llm_records_judges.py
```
