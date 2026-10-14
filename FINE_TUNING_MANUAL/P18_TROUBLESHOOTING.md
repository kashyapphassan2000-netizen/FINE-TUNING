# P18 — Troubleshooting: symptom → cause → exact fix

Find the symptom, apply the fix, re-run the step. Rows marked **(found while writing this manual)** were reproduced on the build machine.

## Laptop / setup

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 1 | `uv` / `git` "is not recognized" | terminal opened before the install | close all PowerShell windows, open a new one; still failing: sign out and in |
| 2 | "running scripts is disabled on this system" | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, `Y`, re-activate `.venv` |
| 3 | `ModuleNotFoundError: peftkit` / `sklearn` / `openpyxl` / `reportlab` | env not active, or deps missing | `.\.venv\Scripts\Activate.ps1`; `uv pip install -r requirements-dev.txt`; `uv pip install -e .` (scikit-learn, openpyxl, python-pptx and reportlab were missing from the original requirements: **found while writing this manual**, now added) |
| 4 | tests: `2 skipped` | they need an NVIDIA GPU | expected |

## Data

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 5 | Excel/CSV rows missing from chunks | old behaviour: every row was its own text, under `min_chars` → dropped **(found, fixed: rows are now grouped)** | update to this repo's version |
| 6 | chunks start mid-word (`"f the relief valve…"`) | character-offset overlap **(found, fixed: overlap starts at a clean boundary)** | update |
| 7 | training rows have system prompt "You write training data… Return JSON" | the generator's instruction was written into the rows **(found, fixed)** | rebuild with this version; pass `--system` |
| 8 | grounded build crashes `KeyError: 'meta'` on a second run | a chunks-only and a grounded build shared one interim folder **(found, fixed: interim per output folder)** | update; or delete `data/interim/` |
| 9 | `ERROR: 0 training rows survived` | near-identical documents / too-long rows | read the printed stage reports; vary documents, raise `--max-tokens` or `--chunk-tokens` |
| 10 | scanned PDF gives no text | no text layer | `ocrmypdf in.pdf out.pdf` |
| 11 | lineage check `BLOCK: proprietary` on your own documents | "proprietary" means someone else's closed data | stamp with `--license owned` **(found, fixed: new licence family)** |
| 12 | synthetic data "utility 0.99" but useless | accuracy on an imbalanced target **(found, fixed: utility uses AUC for binary labels)** | rerun `synthkit.pipeline`; read the AUC |
| 13 | synthetic REJECT on DCR (privacy) | rows too close to real ones | raise `--smoothing` (0.5 passed on the clinic example) |
| 14 | synthetic REJECT on utility, PSI fine | marginals kept, feature → target relationship lost | `--label-mode conditional` (the default now) |

## GPU machine

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 15 | `sudo: command not found` in `setup_gpu_vm.sh` | RunPod/Vast containers run as root without sudo **(found, fixed)** | update the script |
| 16 | `gpu_preflight.sh`: driver supports CUDA 12.x, exit 1 | driver too old for torch 2.14's CUDA-13 wheels | newer image/host; or `TORCH_INDEX=https://download.pytorch.org/whl/cuXXX bash scripts/setup_gpu_vm.sh` after checking that the variant exists (P09) |
| 17 | `torch.cuda.is_available()` False | driver/torch mismatch | row 16 |
| 18 | `No space left on device` | container disk too small / `HF_HOME` on it | put the project and `HF_HOME` on `/workspace` (RunPod/Vast) or `/ephemeral` (HyperStack) |
| 19 | SSH asks for a password | key not on this machine | RunPod/Vast: keys only reach machines created after adding the key; recreate |
| 20 | WinSCP fails on RunPod | proxied SSH (`ssh.runpod.io`) does not support SFTP | use "SSH over exposed TCP" (IP + port) |
| 21 | `scp` "connection refused" on Vast | lower-case `-p` | `scp -P PORT …` (capital P) |
| 22 | `401/403` downloading a model | gated model, licence not accepted, or no token | accept on the model page with the token's account; `hf auth whoami` |

## Training

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 23 | doctor: `generation markers` FAIL | `assistant_only_loss: true` on a template without `{% generation %}` (Qwen, Llama, Mistral official templates **measured: none have them**) | `data.completion_only_loss: true` |
| 24 | `tokenize_inspect` shows 0% trained on an official template | old marker detection matched `add_generation_prompt` **(found, fixed)** | update; use `--completion-only` |
| 25 | `CUDA out of memory` | batch / sequence / method too big | halve `per_device_batch_size`, double `grad_accum`; `max_seq_length` = p95; LoRA → QLoRA; re-run the dry run |
| 26 | loss does not go down | nothing trainable, lr too low (2e-5 on LoRA), everything masked | doctor; `tokenize_inspect`; lr 1e-4 to 5e-4 |
| 27 | loss NaN | lr too high, fp16 overflow | lower lr, `max_grad_norm: 1.0`, bf16 |
| 28 | model never stops talking | end-of-turn token not trained | doctor's `EOS is trained`; fix template/pad |
| 29 | crash mid-run, `--set train.resume=true` did nothing | that key does not exist and is silently ignored **(found, fixed in the toolkit docs)** | add `--resume true` to the same command |
| 30 | multi-GPU loss curve differs from 1-GPU | the global batch changed | `python -m peftkit.batchmath …`; set the printed accumulation |
| 31 | GRPO `grad_norm 0.0`, no learning, after SFT in the same script | the SFT trainer's model object reused **(found while writing lab 25)** | `save_model()`, then reload with `from_pretrained()` |
| 32 | GRPO reward rises, answers get worse | reward hacking | read samples; gate shaping rewards on correctness; tighten `must_match` |

## Evaluation, publishing, serving

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 33 | great test score, poor with real users | test leaked into training, or too similar | `--contamination`/`--eval` guard; human test questions with held-out entities |
| 34 | `bench_gate`: `NO COMMON TASKS` or absurd deltas | base and tuned run with different tasks/limits/shots/template | identical flags; compare the `config` sections |
| 35 | `embedding(): argument 'indices' must be Tensor, not BatchEncoding` | transformers 5 `apply_chat_template` returns a dict | `return_dict=True`, `model.generate(**enc)` (old snippets in the toolkit's F09/F13 chapters **fixed**) |
| 36 | merge diff > tolerance | merged onto a different base/revision | `--base MODEL_ID` with the training revision |
| 37 | served answers worse than offline | template, dtype, `MAX_MODEL_LEN`, adapter name, sampling, or a different `--max-tokens` | `eval_endpoint --offline … --max-tokens` = the offline value |
| 38 | Space build fails or OOMs | model too big for the hardware | small model on cpu-basic; else `--hardware` (paid) or `--backend endpoint` |
| 39 | model card links to a local path | trained from a local folder | `peftkit.card --base-model ORG/NAME` |
| 40 | a big bill | machine left running | P08 Section 8; set a phone alarm every session |
