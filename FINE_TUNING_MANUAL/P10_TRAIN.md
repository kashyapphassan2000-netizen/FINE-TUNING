# P10 — Step 9: Upload the files, configure, check, baseline, dry run, train

**Time:** 30 minutes of setup + the training time from P03. **Cost:** the P03 estimate + 30-50%.
**GOAL:** a trained adapter (or full model) whose run you watched, with a baseline measured **before** training and checkpoints you can pick from.

## The order (never skip a step: each one is cheaper than the failure it prevents)

```text
upload data -> write config -> DOCTOR (0 FAIL) -> BASELINE eval of the untouched model -> 20-step DRY RUN
-> check peak VRAM + speed -> REAL RUN in tmux -> watch loss / eval -> pick the best checkpoint (P11)
```

---

## SECTION 1 — Upload train / eval / test files

**WHERE:** WinSCP (PC left, GPU machine right), or PowerShell `scp`
**GOAL:** the data on the GPU machine at the paths the config uses

Step 1. In WinSCP, drag these from your PC's `FINE-TUNING\peft-universal\` to `WORK/FINE-TUNING/peft-universal/` on the machine:

| From (PC) | To (GPU machine) | What |
|---|---|---|
| `data\processed\train.jsonl`, `eval.jsonl` | `data/processed/` | training data and its held-out split (P06/P07) |
| `data\eval\test.jsonl`, `general.jsonl`, `safety.jsonl` | `data/eval/` | your frozen evaluation suites (P05) |
| `.env` | `.` | already done in P09 |

Or in PowerShell (the `gpu` alias from P08):

```powershell
scp -r data\processed data\eval gpu:WORK/FINE-TUNING/peft-universal/data/
```

Step 2. On the GPU machine, confirm:

```bash
cd WORK/FINE-TUNING/peft-universal && wc -l data/processed/*.jsonl data/eval/*.jsonl
```

**YOU SHOULD SEE:** the same row counts as on your PC.

## SECTION 2 — Write the run config (one YAML per run)

**WHERE:** VS Code (Remote-SSH) → new file `configs/models/SLUG.yaml`
**GOAL:** every decision of this run in one reviewable file

Step 1. Paste, then replace the CAPITALS:

```yaml
run_name: RUN
model:
  name_or_path: MODEL_ID            # Hub id or local folder (P02)
  revision: main                    # better: the exact commit hash you recorded in P02
  dtype: bfloat16                   # float16 on T4/V100 (no bf16)
  attn_implementation: sdpa         # flash_attention_2 only if flash-attn is installed
  max_seq_length: 2048              # >= the p95 token length from P03 Section 2
data:
  train_file: data/processed/train.jsonl
  eval_file: data/processed/eval.jsonl
  completion_only_loss: true        # train ONLY the answer (+ end-of-turn), works with ANY chat template
train:
  output_dir: outputs/RUN
  epochs: 2
  per_device_batch_size: 2
  grad_accum: 8                     # effective batch = 2 x 8 x GPUs = 16
  lr: 2.0e-4                        # LoRA/QLoRA start; DoRA 1e-4; full fine-tuning 1e-5
  warmup_steps: 0.03                # float in [0,1) = ratio of total steps
  scheduler: cosine
  eval_steps: 50
  save_steps: 50
  save_total_limit: 3
  logging_steps: 10
  report_to: tensorboard            # or none / wandb
  seed: 42
```

**Every field explained:**

| Field | Value | Why / when to change |
|---|---|---|
| `max_seq_length` | p95 of your token lengths, rounded up | longer wastes memory; shorter **cuts the answer off** (truncation removes the END) |
| `completion_only_loss: true` | loss on the last assistant turn only | the right default for single-turn data with official templates (Qwen, Llama, Mistral have **no** `{% generation %}` markers, measured in this manual) |
| `assistant_only_loss: true` | loss on **every** assistant turn | only if your template has `{% generation %}` markers (multi-turn data); the doctor FAILs otherwise |
| neither | loss on all tokens | works, less efficient; check `EOS trained` |
| `per_device_batch_size × grad_accum` | 16-64 total | stable gradients; if OOM: halve batch, double accum (same effective batch) |
| `lr` | 2e-4 (LoRA/QLoRA) | **2e-5 is a full fine-tuning lr and is too low for LoRA** |
| `epochs` | 1-3 | more = memorisation; watch eval loss |
| `eval_steps` = `save_steps` | 20-100 | so every saved checkpoint has an eval score |
| `save_total_limit` | 3 | keeps disk bounded (P03 HOST line) |

The method is a separate file: `configs/methods/lora.yaml`, `qlora.yaml`, `dora.yaml`, `adalora.yaml`, `loraplus.yaml`, `full.yaml`. Files are merged left to right: `base.yaml` → method → your model file. Any field can also be overridden on the command line: `--set train.lr=1e-4 method.r=32`.

## SECTION 3 — The doctor (zero FAIL, or do not train)

**WHERE:** GPU terminal, `.venv-train` active, `.env` loaded (P09 Section 3)
**GOAL:** every cheap check before the expensive part

Step 1. Paste and press Enter (use your method file):

```bash
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/SLUG.yaml
```

**YOU SHOULD SEE (shape):**

```text
[PASS] dataset size: 4,812 training examples
[PASS] duplicates in train: 0 exact duplicates
[PASS] train/eval leakage: 0/240 eval examples also appear in train
[PASS] truncation: 0.4% of examples exceed max_seq_length=2048 (p95=1310)
[PASS] chat template: present
[PASS] prompt is a prefix: prompt tokens are an exact prefix of the full conversation
[PASS] loss mask: 21.3% of tokens are trained on (completion-only: last assistant turn)
[PASS] EOS is trained: end-of-turn token in the completion: True
[PASS] trainable params: 43,646,976
[PASS] learning rate: 0.0002 (typical 5e-05-0.001)
[PASS] optimizer steps: 602 planned steps
[PASS] effective batch: 16 sequences per update
12 checks: 12 pass, 0 warn, 0 fail
```

| FAIL / WARN | Fix |
|---|---|
| `train/eval leakage` | remove the overlapping rows (P06 `--eval` guard); otherwise the eval score is meaningless |
| `truncation` > 2% | raise `max_seq_length` to the printed p95, or shorten the examples |
| `generation markers` FAIL | you set `assistant_only_loss` on a template without markers → use `completion_only_loss: true` |
| `loss mask` 0% or ~100% | the prompt/answer split is wrong: run `tokenize_inspect` (below) |
| `EOS is trained: False` | the model will never stop: fix template/pad token |
| `target_modules exist` FAIL | names differ in this architecture: use `all-linear` or the names `peftkit.plan` printed |
| `learning rate` WARN | out of the method's typical range |
| `adalora schedule` FAIL | train longer or lower `tinit_ratio`/`tfinal_ratio` |

Step 2. See exactly what is trained in one row:

```bash
python -m peftkit.tokenize_inspect --tokenizer MODEL_ID --file data/processed/train.jsonl --index 0 --completion-only --limit 80
```

**YOU SHOULD SEE (last line):** `total N tokens, trained on M (x%), EOS trained: True`, and the `YES` rows are exactly the answer.

## SECTION 4 — Baseline: score the untouched model FIRST

**WHERE:** GPU terminal (inside `tmux new -s train`)
**GOAL:** the numbers to beat. Without them, "it improved" is unmeasurable

```bash
tmux new -s train
cd WORK/FINE-TUNING/peft-universal && source .venv-train/bin/activate && set -a; . ./.env; set +a
python -m peftkit.evaluate --model MODEL_ID --suite domain=data/eval/test.jsonl general=data/eval/general.jsonl safety=data/eval/safety.jsonl --max-new-tokens 256 --out reports/base.json
```

**YOU SHOULD SEE:** one line per suite with `n`, `score` (0-1) and `nll`. Write the domain score down: that is the bar.

## SECTION 5 — The 20-step dry run (measure memory and speed before paying for hours)

**WHERE:** same tmux session
**GOAL:** peak VRAM and seconds per step on THIS GPU

Step 1. Paste and press Enter:

```bash
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/SLUG.yaml --set train.max_steps=20 train.output_dir=outputs/dryrun train.eval_steps=1000 train.save_steps=1000
python -c "import json;m=json.load(open('outputs/dryrun/run_meta.json'));print('peak VRAM GB', m.get('peak_vram_gb'), '| seconds', m['train_seconds'], '| s/step', round(m['train_seconds']/m['global_step'],2))"
```

**YOU SHOULD SEE:** `peak VRAM GB 13.4 | seconds 61.2 | s/step 3.06` (your numbers).

| Result | Do this |
|---|---|
| peak VRAM > 90% of the GPU | halve `per_device_batch_size`, double `grad_accum`; or lower `max_seq_length`; or LoRA → QLoRA |
| peak VRAM < 50% | double `per_device_batch_size`, halve `grad_accum` (faster, same maths) |
| s/step × planned steps ≫ the P03 estimate | re-plan with the measured speed (P03 Section 6) before committing |
| `CUDA out of memory` | same as the first row; then re-run this section |

Step 2. Total time = `s/step` × planned optimizer steps (the doctor printed them) + evaluation time. If it is more than you budgeted, stop here and re-plan. This is the cheapest moment to change your mind.

## SECTION 6 — The real run

**WHERE:** same tmux session
**GOAL:** the trained adapter in `outputs/RUN/adapter` (full FT: `outputs/RUN/model`), checkpoints every `save_steps`

Step 1. Paste and press Enter:

```bash
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/SLUG.yaml
```

Step 2. Detach: `Ctrl+b`, then `d`. The run continues. Re-attach: `tmux attach -t train`.

Step 3. Watch it (a second tmux window: `Ctrl+b c`):

```bash
watch -n 5 nvidia-smi                                         # memory stable, GPU-Util high (> 80%)
tensorboard --logdir outputs/RUN --port 6006                  # then on the PC: ssh -L 6006:localhost:6006 gpu
```

**What healthy looks like:**

| Signal | Healthy | Unhealthy → action |
|---|---|---|
| `trainable params` in the first log lines | > 0, the share you expected | 0 → adapter not attached: targets |
| loss, first 50 steps | falling | flat → lr too low, or everything masked (doctor); NaN → lr too high / fp16 overflow: lower lr, bf16 |
| `grad_norm` | bounded (≈0.1-5), no repeated spikes | spikes to 100+ → lower lr, keep `max_grad_norm: 1.0` |
| `eval_loss` every `eval_steps` | falling, then flattening | rising while train loss falls → **overfitting**: the best checkpoint is earlier (P11) |
| GPU utilisation | > 80% | low → data loading or tiny batches |

Step 4. If it crashes or the spot machine is reclaimed, resume from the last checkpoint: **the same command with `--resume true` added to the end**:

```bash
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/SLUG.yaml --resume true
```

**YOU SHOULD SEE at the end:** `{"method": "qlora", "trainable_params": ..., "global_step": ..., "train_loss": ...}` and the files `outputs/RUN/adapter/adapter_model.safetensors`, `outputs/RUN/run_meta.json`, `outputs/RUN/checkpoint-50`, `checkpoint-100`, …

## SECTION 7 — Full fine-tuning (every weight trained)

Use it for small models (≤ 3B) with lots of data, or when LoRA demonstrably under-fits. Memory: 16 B/param (P03), so shard it:

```bash
uv pip install deepspeed==0.19.7
accelerate launch --config_file configs/accelerate/deepspeed_zero3.yaml --num_processes NUM_GPUS -m peftkit.train --config configs/base.yaml configs/methods/full.yaml configs/models/SLUG.yaml
```

The result is a complete model in `outputs/RUN/model` (no merge step). `configs/methods/full.yaml` sets lr 1e-5 (≈10-20× lower than LoRA). Forgetting risk is highest here: replay data (P05 §3) is not optional.

> **WATCH OUT (verify on your machine):** the ZeRO-3 configs were validated by DeepSpeed's and accelerate's own config parsers, but multi-GPU training was not executed while writing this manual. Do the 20-step dry run with the same launch command first, and check that a saved checkpoint loads on one GPU before deleting the machine.

## SECTION 8 — More than one GPU: keep the maths constant

```text
global batch = per_device_batch × grad_accum × GPUs per node × nodes
```

Your tuned lr belongs to a **global** batch. When you add GPUs, keep the global batch and lower the accumulation:

```bash
python -m peftkit.batchmath --global-batch 16 --per-device 2 --gpus-per-node 8 --nodes 1
```

**YOU SHOULD SEE:** `"gradient_accumulation_steps": 1`, `"global_batch_achieved": 16`, `"exact": true`.

| Launch | When | Command |
|---|---|---|
| DDP (data parallel) | the model + training state fit ONE GPU; you want speed | `accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes N -m peftkit.train --config ...` |
| FSDP2 (sharded) | LoRA on a model whose bf16 weights do not fit one GPU | `accelerate launch --config_file configs/accelerate/fsdp2.yaml --num_processes N -m peftkit.train --config ... --set train.gradient_checkpointing=false` |
| DeepSpeed ZeRO-3 | full fine-tuning, or the biggest models | Section 7 |
| 2+ nodes | > 8 GPUs | `configs/accelerate/multinode_fsdp2.yaml`: same command on each node with `--machine_rank 0/1`, `main_process_ip` = node 0's private IP; InfiniBand-class network |
| QLoRA on several GPUs | DDP works (one 4-bit copy per GPU); QLoRA + FSDP has extra constraints | verify the current peft/accelerate docs for your versions before combining them |

Details and the batch-maths table: `peft-universal/docs/manual_src/M07b_distributed.md`.

---

## CHECKPOINT

- [ ] Doctor: 0 FAIL. `tokenize_inspect` shows the answer trained and `EOS trained: True`.
- [ ] `reports/base.json` exists (the baseline).
- [ ] Dry run: peak VRAM < 90% and the time × steps fits my budget.
- [ ] Real run finished; `outputs/RUN/adapter` (or `/model`) and several `checkpoint-*` folders exist.
