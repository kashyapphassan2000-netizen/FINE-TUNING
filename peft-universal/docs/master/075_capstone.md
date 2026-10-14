# 75. Capstone: the unseen-model challenge

:::why The bar
Not "do you know what LoRA is". The bar is: given a causal LM you have never seen and a problem of your own, can you go MODEL → DATA → TOKENIZER → MASKING → PEFT → TRAIN → DEBUG → EVALUATE → IMPROVE → MERGE → DEPLOY without a model-specific tutorial?
:::

## 75.1 The rules

1. Pick a decoder-only causal LM on the Hub that **is not mentioned in either volume** and that you have never fine-tuned. Nobody hands you the name or a config.
2. Use your own data (your documents, your tickets, your notes). At least 300 training examples and a 50-item human-written test set.
3. You may read this book, the code, and the model's own docs. No copy-paste tutorial for that model.
4. Budget: whatever the planner says the smallest viable GPU costs for a few hours. Track the actual spend.

## 75.2 The deliverables

|#|Deliverable|Evidence|
|---|---|---|
|1|compatibility report|`peftkit.compat` output, plus your written reading of its warnings|
|2|resource plan|`peftkit.plan` output and the GPU you chose, with the reason|
|3|dataset + quality report|`reports/quality.json`, and 10 rows you annotated by hand|
|4|tokenization proof|`tokenize_inspect` on one row: trained fraction, EOS trained|
|5|doctor|zero FAIL|
|6|baseline|`reports/base.json` on domain + general suites|
|7|training run|`run_meta.json` (versions, steps, loss, peak VRAM, seconds)|
|8|at least one debugged failure|the symptom, your diagnosis, the fix, the verification|
|9|comparison|`reports/compare.json` with deltas, regressions, verdict|
|10|iteration|failure analysis of 30 items, one fix applied, re-measured|
|11|merge + serving|merge report and a real response from the served model through the gateway|
|12|write-up|one page: what you built, the numbers, what you would do next|

## 75.3 Grading rubric (be strict with yourself)

|Score|What it looks like|
|---|---|
|0-3|it trained, loss went down, no baseline, no test set. **This is where most tutorials stop.**|
|4-5|baseline + test set exist, but the test set overlaps training, or only the domain was measured|
|6-7|clean split, honest before/after, at least one real debugging story, merged and served|
|8-9|+ general/safety suites measured (no silent forgetting), failure analysis drove a second iteration with a measured gain, costs tracked|
|10|+ you could hand the repo and `run_meta.json` to a colleague who reproduces your numbers without asking you a single question|

## 75.4 Difficulty ladder if you want more

1. Same task with a **MoE** base model (choose targets deliberately; see chapter 36).
2. Serve **three domain adapters** on one base with routing and per-key permissions (chapter 71).
3. Add **DPO** on top of your SFT model and show the preference metrics plus unchanged format compliance.
4. Do the whole thing again on a model released after this book was written. If nothing in your process changes, you have learned the thing this book is for.

:::check
- I can name the model, the dataset, and the measured improvement.
- Every failure I hit is written down with its fix.
- Someone else reproduced my result from my repo.
:::
