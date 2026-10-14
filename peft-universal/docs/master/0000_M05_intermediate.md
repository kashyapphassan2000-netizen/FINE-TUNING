# LAB MANUAL — PART 5: INTERMEDIATE — Your data, a real model, a rented GPU

:::why Concept (what, why, when)
Same loop as Part 4, now with a real open model, your own documents, and a rented GPU. **Why:** this is the job as companies give it: "here are our documents, make the model good at our work". **When:** after Part 4 passes on your laptop.
:::

:::honest What was and was not executed
The laptop steps (openness, planning, data build, doctor) were run while writing this manual. The GPU steps were checked against the source code but not executed here (no GPU in the build machine); expected outputs describe what to look for, not captured text.
:::

### Placeholders used in this Part

|Placeholder|Replace with|Example|
|---|---|---|
|MODEL_ID|the Hugging Face id of the base model|meta-llama/Llama-3.1-8B-Instruct|
|YOUR_DOMAIN|a short description of the domain|industrial pump support|
|VM_IP|the public IP shown in the Hyperstack console|203.0.113.10|
|RUN|the run folder name you choose|pump-sft|

## SECTION 1 — Check the model is open and usable

**WHERE:** PowerShell, `(.venv)` active

**GOAL:** a PROCEED decision, with the licence recorded

Step 1. Type and press Enter:

```laptop
python -m peftkit.openness MODEL_ID --intended-use commercial
```

**You should see:** a JSON block with `"decision": "PROCEED"`, `"STOP"` or `"UNKNOWN"`.

|If you see…|It means…|Do this next|
|---|---|---|
|`STOP` … not open-weights|no downloadable weights|pick another model|
|`STOP` … not permit commercial use|licence forbids your use|pick another model, or change intended use|
|GATED in actions|you must accept terms|open the model page in the browser, click the accept button while logged in|
|`UNKNOWN` … could not read the Hub|offline, no token, or a typo in MODEL_ID — it has NOT judged the licence|set `HF_TOKEN` in `.env`, check the id, retry|

:::check
- Decision PROCEED; licence family and duties noted in your project notes.
:::

## SECTION 2 — Work out the hardware and cost

**WHERE:** PowerShell

**GOAL:** know the GPU type and count before renting

Step 1. Type and press Enter:

```laptop
python -m peftkit.plan MODEL_ID --seq-len 4096 --batch 1
```

**You should see:** a table of memory per method (lora, qlora, …) and which GPUs fit.

Step 2. Write down the smallest GPU whose memory fits **QLoRA** with ~30% headroom.

:::warn Watch out
The plan is an estimate. Section 8 (the dry run) confirms it before you pay for hours.
:::

:::check
- You know: GPU type, GPU count, and whether QLoRA or LoRA.
:::

## SECTION 3 — Build the dataset from your documents

**WHERE:** File Explorer, then PowerShell

**GOAL:** `data\processed\train.jsonl` and `eval.jsonl`, leak-free

Step 1. Copy your documents (PDF, DOCX, TXT, MD, CSV, HTML, JSON) into `data\raw`.

Step 2. In VS Code, create `data\eval\test.jsonl` with 50-200 real questions and expected answers written by a person, one per line, in the same shape as `data\lesson\test.jsonl`.

Step 3. Type and press Enter (grounded mode calls your LLM API from `.env`; use `--mode chunks-only` to avoid API calls):

```laptop
python -m peftkit.data.build --docs data/raw --out data/processed --tokenizer MODEL_ID --domain "YOUR_DOMAIN" --mode grounded --eval data/eval/test.jsonl --max-chunks 2000 --per-chunk 3
```

**You should see:** four stages (`1/4 ingest` … `4/4 format`) and `ready: data/processed/train.jsonl (N rows)`.

|If you see…|It means…|Do this next|
|---|---|---|
|`ERROR: 0 training rows survived` with `drop_repetitive`|your documents are near-identical boilerplate|add varied documents, or raise `--chunk-tokens`|
|`too_long` large in the format line|examples exceed `--max-tokens`|raise `--max-tokens` to your model's context, or chunk smaller|
|`eval.jsonl (0 rows)` on a small corpus|the split rounded down|fine: your real test set is `data/eval/test.jsonl`|

Step 4. Type and press Enter:

```laptop
python -m peftkit.tokenize_inspect --tokenizer MODEL_ID --file data/processed/train.jsonl --index 0
```

Step 5. Open `data\processed\train.jsonl` and read 30 random lines.

:::warn Watch out
If you read 30 lines and find wrong answers, stop and fix the generation. Training on wrong answers teaches wrong answers.
:::

:::check
- `train.jsonl` and `eval.jsonl` exist; the trained fraction is sensible; 30 lines read.
:::

## SECTION 4 — Write the model config and run the doctor

**WHERE:** VS Code, then PowerShell

**GOAL:** a config for your run, and zero FAIL

Step 1. In VS Code, create `configs\models\my-model.yaml` with:

```yaml configs/models/my-model.yaml
run_name: RUN
model:
  name_or_path: MODEL_ID
  max_seq_length: 4096
data:
  train_file: data/processed/train.jsonl
  eval_file: data/processed/eval.jsonl
  assistant_only_loss: true
train:
  output_dir: outputs/RUN
  epochs: 2
  per_device_batch_size: 2
  grad_accum: 8
  lr: 2.0e-4
  eval_steps: 50
  save_steps: 50
  save_total_limit: 3
```

|Field|Value|Why|
|---|---|---|
|max_seq_length|the p95 token length from Step 4 above, rounded up|longer wastes memory; shorter cuts answers|
|per_device_batch_size × grad_accum|16|a stable effective batch|
|lr|2e-4|the LoRA/QLoRA starting point (2e-5 is for full fine-tuning and is too low)|
|eval_steps / save_steps|50|so you can pick the best checkpoint later|

:::tip Under the hood (standard libraries)
These YAML fields become `trl.SFTConfig(max_length=..., per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4, eval_steps=50, save_steps=50, save_total_limit=3, assistant_only_loss=True)`; `configs/methods/qlora.yaml` adds `transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`. See Part 5B.
:::

Step 2. Press `Ctrl+S`.

Step 3. Type and press Enter:

```laptop
python -m peftkit.diagnostics --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
```

:::check
- `0 fail`. Commit and push (`git add .`, `git commit -m "data + config"`, `git push`).
:::

## SECTION 5 — Rent the GPU

**WHERE:** browser, https://console.hyperstack.cloud

**GOAL:** a running Ubuntu VM with the GPU from Section 2

Step 1. Click **Billing** and add a small credit.

Step 2. Click **Environments** → **Create Environment**, choose a region, click **Create**.

Step 3. Click **SSH Keys** → **Import**. (Create the key first: next section, Steps 1-2, then return here and paste the contents of the `.pub` file.)

Step 4. Click **Virtual Machines** → **Deploy New Virtual Machine**.

Step 5. Fill the form:

|Field|Value|Why|
|---|---|---|
|Environment|the one from Step 2|region and network|
|Flavor / GPU|the GPU from Section 2 (e.g. 1× L40 48 GB, or 1× A100 80 GB)|fits QLoRA with headroom|
|Image|an Ubuntu image **with NVIDIA drivers** new enough for CUDA 13 (verify in the image description)|torch 2.14 wheels need it|
|SSH key|the key you imported|passwordless login|
|Public IP|enabled|so you can connect|

Step 6. Click **Deploy**.

Step 7. Wait until the status is **ACTIVE**, then copy the **public IP** (this is VM_IP).

:::warn Watch out: billing
Billing starts when the VM is ACTIVE, not when training starts. Hyperstack's on-demand H100 PCIe was listed from $2.50/h and spot from $2.00/h when checked [3][4]; always read the price on the deploy screen.
:::

:::check
- VM status ACTIVE and you have VM_IP.
:::

## SECTION 6 — Connect from Windows

**WHERE:** PowerShell, VS Code, WinSCP

**GOAL:** terminal, editor and file transfer all reaching the VM

:::tip New to Linux?
Do Part 4B (WSL 2) first: the bash commands, tmux and VS Code's Linux mode you use below are practised there on your own laptop.
:::

Step 1. Type and press Enter (press Enter again at both prompts for no passphrase, or set one):

```laptop
ssh-keygen -t ed25519 -f $HOME\.ssh\hyperstack
```

Step 2. Type and press Enter, then paste the printed line into Hyperstack's SSH key import (Section 5 Step 3):

```laptop
Get-Content $HOME\.ssh\hyperstack.pub
```

Step 3. Type and press Enter to open the SSH config file:

```laptop
notepad $HOME\.ssh\config
```

Step 4. Paste these lines, replace VM_IP, save, close Notepad:

```text C:\Users\YOUR_NAME\.ssh\config
Host hs-train
    HostName VM_IP
    User ubuntu
    IdentityFile ~/.ssh/hyperstack
    ServerAliveInterval 30
```

Step 5. Type and press Enter; answer `yes` to the fingerprint question:

```laptop
ssh hs-train
```

**You should see:** a Linux prompt like `ubuntu@...:~$`.

Step 6. Type `exit` and press Enter.

Step 7. In VS Code press `F1`, type `Remote-SSH: Connect to Host`, press Enter, choose `hs-train`.

Step 8. Open WinSCP → **New Session** → fill: File protocol **SFTP**, Host name **VM_IP**, User name **ubuntu** → **Advanced...** → **SSH > Authentication** → Private key file: choose `hyperstack` (WinSCP offers to convert it to .ppk — accept) → **OK** → **Login**.

:::check
- `ssh hs-train` logs in with no password; VS Code shows "SSH: hs-train" bottom-left; WinSCP shows the VM's home folder.
:::

## SECTION 7 — Put the project on the VM and install the stack

**WHERE:** VS Code terminal connected to hs-train (bash)

**GOAL:** GPU verified, two environments created

Step 1. Type and press Enter:

```gpu
git clone https://github.com/YOUR_GITHUB_USER/peft-universal.git
```

Step 2. Type and press Enter:

```gpu
cd peft-universal
```

Step 3. In WinSCP, drag your local `data\processed`, `data\eval` and `.env` into the VM's `peft-universal` folder (they are not in GitHub).

Step 4. Type and press Enter:

```gpu
bash scripts/gpu_preflight.sh
```

**You should see:** an OK line about the driver supporting CUDA 13. If it stops, the image's driver is too old — redeploy with a newer image.

Step 5. Type and press Enter:

```gpu
python3 -m peftkit.gpu_stack
```

(If `peftkit` is not importable yet, run it after Step 6.)

Step 6. Type and press Enter:

```gpu
bash scripts/setup_gpu_vm.sh
```

Step 7. Type and press Enter:

```gpu
source .venv-train/bin/activate
```

Step 8. Type and press Enter:

```gpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**You should see:** `2.14.0+cu130 True` and your GPU's name.

:::check
- `True` and the right GPU name. If False: driver/CUDA mismatch — read Reference chapter 55's preflight notes.
:::

## SECTION 8 — Baseline, dry run, real run

**WHERE:** VS Code terminal on hs-train, inside `tmux`

**GOAL:** a trained adapter, with the run surviving a dropped connection

Step 1. Type and press Enter:

```gpu
tmux new -s train
```

Step 2. Type and press Enter:

```gpu
source .venv-train/bin/activate
```

Step 3. Type and press Enter (the baseline):

```gpu
python -m peftkit.evaluate --model MODEL_ID --suite domain=data/eval/test.jsonl --out reports/base.json
```

Step 4. Type and press Enter (20-step dry run):

```gpu
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml --set train.max_steps=20 train.output_dir=outputs/dryrun
```

Step 5. Type and press Enter:

```gpu
python -c "import json;m=json.load(open('outputs/dryrun/run_meta.json'));print(m['peak_vram_gb'],'GB', m['train_seconds'],'s')"
```

**You should see:** a peak VRAM below your GPU's memory. If it is near the limit, halve `per_device_batch_size` and double `grad_accum`.

Step 6. Type and press Enter (the real run):

(More than one GPU? Read Part 7B first: the accumulation must change so the global batch stays the same.)

```gpu
python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml
```

Step 7. Press `Ctrl+b`, then `d`, to detach. Training continues.

Step 8. To come back later, type and press Enter:

```gpu
tmux attach -t train
```

:::warn Watch out
If the run crashes, rerun Step 6 with `--resume true` added to the end of the command, to continue from the last checkpoint.
:::

:::check
- `outputs/RUN/adapter` exists and checkpoints `checkpoint-50`, `checkpoint-100`, … exist.
:::

## SECTION 9 — Evaluate and compare

**WHERE:** VS Code terminal on hs-train

**GOAL:** a PASS verdict, and the best checkpoint identified

Step 1. Type and press Enter:

```gpu
python -m peftkit.evaluate --model outputs/RUN/adapter --suite domain=data/eval/test.jsonl --out reports/tuned.json
```

Step 2. Type and press Enter:

```gpu
python -m peftkit.evaluate --compare reports/base.json reports/tuned.json --out reports/compare.json
```

**You should see:** a before/after table and `verdict: PASS …`.

Step 3. Repeat Step 1 with `--model outputs/RUN/checkpoint-50`, `-100`, … and note which scores best.

:::check
- Verdict PASS, and you know the best checkpoint number.
- If FAIL: go to Part 10 (Troubleshooting) rows 5-7 before training again.
:::

## 5.10 Common mistake at this level

**Testing on data that looks like training data.** If `data/eval/test.jsonl` was generated from the same documents with the same generator, a high score only proves the model memorised your generator's style. **How to spot it:** scores that are near-perfect on your test set but poor when a colleague asks questions in their own words. Fix: human-written test questions, held-out entities, and the `--eval` contamination check in Section 3.
