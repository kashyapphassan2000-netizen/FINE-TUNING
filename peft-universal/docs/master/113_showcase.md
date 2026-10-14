# 113. Showcasing the work to owners, customers or the public

:::why In one line
Show evidence, not claims: what was trained, how it compares with the base model on unseen tests, what got worse, whether the deployed API matches, how much load one server takes, and what it costs — all read from files your pipeline produced.
:::

@file src/peftkit/showcase.py

```any
python -m peftkit.showcase --title "Pump support assistant" \
  --compare reports/compare.json --tuned reports/tuned.json --parity reports/parity.json \
  --loadtest reports/loadtest.json --capacity reports/capacity.json --run outputs/<run>/run_meta.json \
  --out reports/showcase.html
```

The report contains: the training facts (base model, method, trainable parameters, steps, hardware, library versions), before/after per suite with the verdict, **the items that got worse**, sample answers, offline-vs-served parity, the load-test table and per-replica capacity, and the capacity/cost table with the Little's-law line — plus an "honest limits" section. The rehearsal builds one from real files (chapter 2, step 14).

## 113.1 What convinces a technical audience

1. The baseline, measured on the same suites.
2. Regressions shown, not hidden.
3. The served API matching offline results.
4. Capacity and cost from measurements, with the formulas.
5. A reproducible repository: someone else runs the rehearsal and the runbook and gets the same numbers.

## 113.2 What to put on a public page or a CV

Placeholders only until you have your own numbers: "[model] fine-tuned with [method] on [N] examples; domain score [A]→[B] on [M] unseen items; general suite [C]→[D]; served at p95 [L] s up to [K] concurrent requests per [GPU]; [$X] per 1M output tokens at [price]/GPU-hour."
