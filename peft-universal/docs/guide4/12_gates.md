# 12. Step 9: quality gates and the scorecard

@file src/synthkit/gates.py

## 12.1 Ten gates, hard and soft

|#|Gate|Hard?|Fails when|
|---|---|---|---|
|1|schema|yes|any contract violation|
|2|constraints|yes|any business rule broken|
|3|excess duplicates|no|more repetition than the real data itself|
|4|privacy: exact match|yes|real records copied|
|5|privacy: DCR ratio|yes|synthetic sits on top of real|
|6|privacy: membership inference|yes|AUC > 0.6|
|7|PII|yes|any personal data in output|
|8|distribution|no|max PSI above the contract threshold|
|9|combination coverage|no|real category combinations missing|
|10|utility (TSTR/TRTR)|yes|synthetic does not help on real data|
|11|judge pass rate|no|below 80% (when a judge is used)|

**One hard failure = REJECT the batch.** "The average score is fine" is how bad data ships.

## 12.2 The decision this produces

@lab lab40_end_to_end

The top table is the whole argument of this book in six numbers: the same pipeline, one parameter changed. With `smoothing=0` the data is faithful (PSI 0.070) and useful (utility 0.98) — and it is **REJECTED**, because it copies real values: exact-match rate 0.242, DCR ratio 0.106, membership-inference AUC 0.834. With `smoothing=0.25` fidelity and utility are unchanged (PSI 0.061, utility 1.04) and every privacy gate passes, so it **SHIPS**.

Without gates, the first dataset would have shipped. It looked excellent on every metric a normal project measures.

## 12.3 The scorecard

Six axes, 0-100, with transparent formulas in the source: fidelity, coverage, diversity, utility, privacy, cleanliness. Use it to compare **versions of your own dataset** over time. Do not compare scorecards across projects: the thresholds are yours.

Ship only when: every hard gate passes **and** utility beats your minimum **and** privacy passes **and** there is a measured improvement on the real gold test.

:::check
- Gate thresholds live in the contract, under version control.
- A rejected batch is archived with its report, not deleted (you will want the counter-example).
- The scorecard is tracked per dataset version.
:::
