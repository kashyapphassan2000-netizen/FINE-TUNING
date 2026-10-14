# LAB MANUAL — PART 10: Troubleshooting

|#|Symptom|Likely cause|Exact fix (steps)|
|---|---|---|---|
|1|`uv` or `git` "is not recognized"|terminal opened before the install|1. close all PowerShell windows 2. open a new one 3. retry; if still failing, sign out and in|
|2|"running scripts is disabled on this system"|PowerShell execution policy|1. `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 2. answer `Y` 3. re-run `.\.venv\Scripts\Activate.ps1`|
|3|`ModuleNotFoundError: No module named 'peftkit'`|environment not active, or project not installed|1. `.\.venv\Scripts\Activate.ps1` 2. `uv pip install -e .` 3. retry|
|4|`torch.cuda.is_available()` is False on the VM|driver too old for the torch CUDA build|1. `bash scripts/gpu_preflight.sh` 2. if it fails, redeploy with an image whose driver supports CUDA 13 3. rerun setup|
|5|`CUDA out of memory`|batch, sequence length or method too large for the GPU|1. halve `per_device_batch_size`, double `grad_accum` 2. lower `max_seq_length` to the p95 3. switch LoRA → QLoRA 4. re-run the 20-step dry run|
|6|loss does not go down|nothing trainable, lr too low, or everything masked|1. check `trainable_params` > 0 in the log 2. `tokenize_inspect` shows 2-90% trained 3. lr 1e-4…5e-4|
|7|loss is NaN|lr too high, fp16 overflow, broken labels|1. lower lr 2. `max_grad_norm: 1.0` 3. bf16 instead of fp16 4. inspect one batch|
|8|model never stops talking|end-of-turn token not trained|1. `tokenize_inspect` → `EOS trained` must be True 2. fix the chat template/pad token 3. retrain|
|9|great score on your test, poor with real users|test set leaked or too similar to training|1. `peftkit.data.quality --contamination data/eval/test.jsonl` 2. write human test questions with held-out entities 3. re-evaluate|
|10|served answers worse than offline|serving differs from training|1. run `peftkit.eval_endpoint --offline reports/tuned.json` 2. check template, dtype, `MAX_MODEL_LEN`, adapter name, sampling in that order|
|11|`wsl --install` fails or VERSION shows 1|virtualization off, or WSL 1|1. enable VT-x/AMD-V in BIOS 2. `wsl --update` 3. `wsl --set-version Ubuntu-24.04 2`|
|12|`embedding(): argument 'indices' must be Tensor, not BatchEncoding`|transformers 5 `apply_chat_template` returns a dict-like object|1. add `return_dict=True` 2. use `["input_ids"]` from the result|
|13|multi-GPU run's loss curve differs from the single-GPU run|global batch changed with the GPU count|1. `python -m peftkit.batchmath --global-batch G --per-device B --gpus-per-node N --nodes M` 2. set the printed accumulation 3. re-check warmup/eval/save steps|
|14|`bench_gate` says `NO COMMON TASKS` or deltas look absurd|base and tuned benchmarked with different tasks, limits, shots or templates|1. rerun both with identical `--tasks`, `--limit`, `--apply_chat_template` 2. compare the `config` sections of both results files|
|15|`nginx: unknown directive "http2"` or certbot cannot connect|nginx older than 1.25.1, or port 80 closed|1. use `listen 443 ssl http2;` (as shipped) 2. open port 80 and check the DNS A record|

The full symptom → cause → command → expected → fix → verify table is Reference chapter 102.
