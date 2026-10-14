# 1. What was missing, and what this edition adds

:::why Why this chapter exists
You asked for a brutally honest check. Here it is, measured against the actual files of the two source books, not against memory.
:::

## 1.1 Gaps found in the Universal PEFT Build Guide

|#|Gap|How serious|Fixed in|
|---|---|---|---|
|1|**No evaluation after training.** The build path went train → merge → a single sanity `generate` → publish. `peftkit.evaluate` existed only in the second book and was never called in the build guide.|Critical: you could ship a model that got worse|chapters 44-45 in the build order, runbook steps 30-33|
|2|**No baseline before training.** Without the base model's score on the same suites, "it improved" is unmeasurable.|Critical|runbook step 26|
|3|**No evaluation of the SERVED model.** vLLM can change the chat template, dtype, quantization, sampling defaults, context length and which adapter answers. None of that was tested.|High: silent quality loss in production|new chapter 53 + `peftkit.eval_endpoint`|
|4|**No load test.** The autoscaling threshold ("queue depth 8") was a guess. Nothing measured latency under concurrency.|High: wrong capacity, wrong bill|new chapter 54 + `peftkit.loadtest`|
|5|**No release gate** linking quality → serving parity → capacity before scaling.|High|new chapter 55 + `peftkit.release_gate`|
|6|**No end-to-end rehearsal.** Each stage was tested alone; the chain was never run as one.|Medium: integration bugs appear on the paid GPU|`scripts/rehearsal_cpu.py`, runbook step 12|
|7|Several `.sh` scripts cannot run in plain PowerShell.|Medium for Windows beginners|the rehearsal is Python; the runbook marks every bash step and gives the WSL route|
|8|Best-checkpoint selection existed only in the second book; the build guide merged the last checkpoint.|Medium|runbook step 34|

## 1.2 Gaps in Fine-Tuning From Zero

|#|Gap|Fixed in|
|---|---|---|
|1|It evaluated models offline only — never through the API users hit.|chapter 53|
|2|It stopped at "merged and served"; no capacity or scaling method.|chapters 54-56|
|3|It assumed the build guide for setup; a beginner had to juggle two books.|merged here in execution order|

## 1.3 Still not solved, stated plainly

- **Nothing here was run on a real GPU or a real pretrained model.** The rehearsal proves the *plumbing*; it cannot prove quality on your 8B model. You still run the same commands on the GPU and read the numbers.
- **The local server (`serve_local`) handles one request at a time.** The rehearsal's load test shows it: p50 latency roughly doubles per concurrency level while throughput stays flat around 20 requests/s. It is for rehearsal; production uses vLLM, and you must re-run the load test against vLLM because those numbers are the ones that set your autoscaling.
- **The plain-VM autoscaler's delete call is still unverified** (the Hyperstack reference page is JavaScript-rendered). It only runs with `--allow-delete`. Kubernetes + KEDA is the recommended path.
- **No differential privacy, no GPU benchmark, no prices** anywhere in this book.

:::check
- I know the three post-training stages the original build guide skipped: served-model evaluation, load testing, release gating.
- I will run the rehearsal before renting a GPU.
:::
