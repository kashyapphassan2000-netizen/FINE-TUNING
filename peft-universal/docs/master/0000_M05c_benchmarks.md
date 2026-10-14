# LAB MANUAL — PART 5C: Benchmarks — prove nothing general was forgotten (before the release gate)

:::why Concept (what, why, when)
Your own `test.jsonl` proves the **domain** improved. Public benchmarks prove the model did **not lose** general ability — maths, knowledge, instruction following — which narrow fine-tuning can quietly destroy ("catastrophic forgetting"). **Why:** a support bot that can no longer do arithmetic on invoice totals is a regression your domain test never asked about. **When:** after Part 5's PASS verdict, before Part 6's release gate.
:::

:::honest What was and was not executed
lm-evaluation-harness was **not run** here: its tasks download datasets from the Hub (unreachable from the build machine) and realistic runs need a GPU. Everything below was read from **lm-eval 0.4.13's source**: the flags, the `peft=` argument of the `hf` backend, the results file name `results_<date>.json`, and the metric keys such as `acc,none` and `exact_match,strict-match`. The comparison tool (`peftkit.bench_gate`) is tested on files in exactly that format.
:::

## 5C.1 Which benchmark measures what

|Task name in lm-eval|Measures|Type|Can run through a chat API?|
|---|---|---|---|
|`mmlu`|broad knowledge, multiple choice|loglikelihood|**no** — needs `vllm` or `hf` backend|
|`hellaswag`, `arc_challenge`, `winogrande`|commonsense / reasoning, multiple choice|loglikelihood|**no**|
|`gsm8k`|grade-school maths, generated answers|generation|yes|
|`ifeval`|instruction following (format constraints)|generation|yes|

The "no" is a fact from the source: lm-eval's chat-completions backend raises `NotImplementedError("Loglikelihood is not supported for chat completions...")`.

## SECTION 1 — Install lm-eval in its own environment

**WHERE:** GPU VM (bash)

**GOAL:** lm-eval available without touching the training or serving environments

Step 1. Type and press Enter:

```gpu
uv venv .venv-eval --python 3.11
```

Step 2. Type and press Enter:

```gpu
source .venv-eval/bin/activate
```

Step 3. Type and press Enter:

```gpu
uv pip install "lm-eval[vllm]==0.4.13"
```

Step 4. Type and press Enter:

```gpu
lm_eval --help | head -5
```

**You should see:** the usage line of `lm_eval`.

:::check
- `lm_eval --help` works inside `.venv-eval`.
:::

## SECTION 2 — Benchmark the BASE model

**WHERE:** GPU VM, `.venv-eval`, inside tmux

**GOAL:** the general-ability numbers to protect

Step 1. Type and press Enter (replace MODEL_ID):

```gpu
lm_eval --model vllm --model_args "pretrained=MODEL_ID,dtype=bfloat16,gpu_memory_utilization=0.85" --tasks mmlu,gsm8k --batch_size auto --apply_chat_template --output_path reports/benchmarks/base
```

**You should see:** a results table per task, and a file `reports/benchmarks/base/<model>/results_<date>.json`.

:::warn Watch out: cost and time
Full MMLU is thousands of questions. For a first pass add `--limit 200` to every command in this Part — and use the **same** limit for base and tuned, or the comparison is meaningless.
:::

## SECTION 3 — Benchmark the FINE-TUNED model

**WHERE:** same

**GOAL:** the same numbers for your model — either the adapter (before merging) or the merged model

Option A — the adapter, no merge needed (the `hf` backend accepts `peft=`):

```gpu
lm_eval --model hf --model_args "pretrained=MODEL_ID,peft=outputs/RUN/checkpoint-BEST,dtype=bfloat16" --tasks mmlu,gsm8k --batch_size auto --apply_chat_template --output_path reports/benchmarks/tuned
```

Option B — the merged model (faster, with vLLM):

```gpu
lm_eval --model vllm --model_args "pretrained=outputs/RUN/merged,dtype=bfloat16,gpu_memory_utilization=0.85" --tasks mmlu,gsm8k --batch_size auto --apply_chat_template --output_path reports/benchmarks/tuned
```

:::check
- `reports/benchmarks/tuned/<model>/results_<date>.json` exists, run with the same tasks and limit as the base.
:::

## SECTION 4 — Gate on forgetting

**WHERE:** same

**GOAL:** PASS only if no benchmark dropped by more than your tolerance

@file src/peftkit/bench_gate.py

Step 1. Type and press Enter:

```gpu
source .venv-train/bin/activate
```

Step 2. Type and press Enter:

```gpu
python -m peftkit.bench_gate --base reports/benchmarks/base --tuned reports/benchmarks/tuned --max-drop 0.02 --out reports/bench_gate.json
```

**You should see** a table like this (format from the test suite, which uses lm-eval's exact result layout):

```out
task                  metric                            base   tuned    delta
mmlu                  acc,none                          0.62   0.615  -0.0050
gsm8k                 exact_match,strict-match          0.55    0.49  -0.0600  <-- DROP
hellaswag             acc_norm,none                     0.79   0.795  +0.0050
verdict: FAIL
```

|If you see…|It means…|Do this next|
|---|---|---|
|verdict PASS|general ability held within 2 points|continue to Step 3|
|a DROP on one task|forgetting on that skill|add 5-30% general/replay data, fewer epochs or lower lr/rank, retrain, rerun Sections 3-4|
|`NO COMMON TASKS`|base and tuned were run on different tasks|rerun with identical `--tasks` and `--limit`|

Step 3. Feed the result into the release gate (Part 6, Section 6) with the optional fourth check:

```gpu
python -m peftkit.release_gate --compare reports/compare.json --parity reports/parity.json --loadtest reports/loadtest.json --bench reports/bench_gate.json
```

**You should see:** a fourth line `[PASS] benchmarks PASS` — or `BLOCK` naming the dropped task.

:::check
- `reports/bench_gate.json` says PASS, and the release gate shows four PASS lines.
:::

## 5C.5 Common mistake at this level

**Comparing numbers that were measured differently.** Base with 5-shot and tuned with 0-shot, chat template on one and off the other, `--limit 200` on one and the full set on the other. The difference then measures your settings, not your model. **How to spot it:** open both `results_*.json` files and compare their `config` sections before believing any delta.
