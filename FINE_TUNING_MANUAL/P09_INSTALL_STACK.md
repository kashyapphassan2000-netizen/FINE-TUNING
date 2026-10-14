# P09 — Step 8: Install the right stack for THIS GPU, then load the model and prove it runs

**Time:** 20-45 minutes. **Cost:** GPU time (keep it short: everything here was rehearsed in P01).
**GOAL:** driver ↔ CUDA ↔ torch matched, two isolated Python environments (training, serving), your model downloaded, loaded on the GPU, generating text, with peak VRAM measured.

## Why the order matters

```text
NVIDIA driver (on the machine)  supports CUDA up to version X        <- you cannot change this on RunPod/Vast
  torch wheel built for CUDA Y   must have Y <= X                     <- the #1 first-day failure
    transformers / peft / trl / bitsandbytes on top of torch
      vLLM pins its OWN torch version -> it gets its OWN environment (.venv-serve)
```

The pinned training stack is torch 2.14.0, whose PyPI wheels are CUDA 13.0 builds → the driver must support CUDA ≥ 13 (driver R580+).

---

## SECTION 1 — Check the machine before installing anything

**WHERE:** VS Code terminal connected to `gpu` (bash)
**GOAL:** know the GPU, driver, CUDA limit, disk and RAM, in 10 seconds

Step 1. Paste and press Enter (replace `WORK` and `REPO_BRANCH`):

```bash
cd WORK
git clone -b REPO_BRANCH https://github.com/kashyapphassan2000-netizen/FINE-TUNING.git
cd FINE-TUNING/peft-universal
bash scripts/gpu_preflight.sh
```

**YOU SHOULD SEE (shape):**

```text
== GPU
name, memory.total [MiB], driver_version
NVIDIA H100 80GB HBM3, 81559 MiB, 580.xx.xx
driver major=580  max CUDA supported by driver=13.0
== CPU/RAM/disk
<cores> / <RAM> / <disk free>
== NVLink / topology
...
OK: driver can run CUDA 13 wheels
```

| If you see… | It means… | Do this |
|---|---|---|
| `WARNING: driver supports CUDA 12.x` and exit 1 | driver too old for the default torch wheel | best: choose another machine/image (HyperStack: newer image; Vast: filter "CUDA ≥ 13"). Or install a torch build for your CUDA: check which CUDA variants exist for torch 2.14 on https://pytorch.org/get-started/locally/, then use `TORCH_INDEX=https://download.pytorch.org/whl/cuXXX` in Section 2 (verify) |
| `nvidia-smi: command not found` | no driver / not a GPU machine / wrong image | redeploy with a GPU image |
| free disk smaller than P03's HOST line | weights will not fit | add a volume or redeploy with a bigger disk |
| fewer GPUs than you paid for | wrong flavor | check the console before training |

Step 2. Ask the toolkit what this GPU supports (bf16, fp8, FlashAttention) and the exact install plan:

```bash
PYTHONPATH=src python3 -m peftkit.gpu_stack
```

(It needs no installed packages, so it works before Section 2. Verified with a bare system Python.)

**YOU SHOULD SEE:** `GPU: H100  spec: {'cc': 9.0, 'vram': 80, 'bf16': True, 'fp8': True, 'flash_attn': True, ...}`, the 8 numbered install steps, and WARNING lines such as "T4 has no bf16: use fp16" when they apply.

## SECTION 2 — Install both environments (one script)

**WHERE:** same terminal, in `FINE-TUNING/peft-universal`
**GOAL:** `.venv-train` (torch 2.14 + transformers/peft/trl/bitsandbytes) and `.venv-serve` (vLLM 0.29)

Step 1. Start tmux first (the install takes 5-15 minutes; a dropped connection must not kill it):

```bash
tmux new -s setup
```

Step 2. Paste and press Enter (works on HyperStack with `sudo` and on RunPod/Vast.ai as root):

```bash
bash scripts/setup_gpu_vm.sh
```

Older driver (Section 1 table)? Use the matching torch build instead (verify the index exists first):

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_gpu_vm.sh
```

**YOU SHOULD SEE (last lines):**

```text
train env OK | torch 2.14.0+cu130 | cuda available: True | gpus: 1 | bf16: True
peft 0.21.0 transformers 5.17.0 trl 1.13.0 bnb 0.50.2
serve env OK | vllm 0.29.0 | torch 2.13.0+...
```

| If you see… | It means… | Do this |
|---|---|---|
| `cuda available: False` | driver/torch CUDA mismatch | Section 1 table, first row |
| `bf16: False` | T4/V100-class GPU | set `model.dtype: float16` in your config (P10); expect less stability |
| bitsandbytes import error | no CUDA visible at import, or an unsupported GPU | QLoRA needs NVIDIA CUDA; use LoRA if bnb cannot load |
| vLLM install fails | a vLLM build for this CUDA/GPU is not available | you only need vLLM for serving (P15); training is unaffected |
| `No space left on device` | small container disk (RunPod) | put the project and `HF_HOME` on `/workspace` (the volume) |

Step 3. Detach (`Ctrl+b`, then `d`). Re-attach later with `tmux attach -t setup`.

## SECTION 3 — Your secrets and caches on the GPU machine

**WHERE:** WinSCP (PC) and the GPU terminal
**GOAL:** `.env` on the machine, the model cache on the big disk, Hugging Face logged in

Step 1. In WinSCP drag your PC's `FINE-TUNING\peft-universal\.env` into `WORK/FINE-TUNING/peft-universal/` on the machine.

Step 2. Edit `HF_HOME` in that `.env` for this provider: HyperStack `/ephemeral/hf-cache`; RunPod/Vast.ai `/workspace/hf-cache`.

Step 3. Paste and press Enter:

```bash
cd WORK/FINE-TUNING/peft-universal
source .venv-train/bin/activate
set -a; . ./.env; set +a
hf auth whoami
```

**YOU SHOULD SEE:** your `HF_USER`. (`hf auth login` and paste the token if not.)

> **WATCH OUT:** `set -a; . ./.env; set +a` loads the variables only into **this** terminal. Repeat it in every new terminal/tmux window, or the token and `HF_HOME` are missing.

## SECTION 4 — Download the model (and time it)

**WHERE:** GPU terminal, `.venv-train` active, `.env` loaded
**GOAL:** weights in `HF_HOME`, the download time known (it is your cold-start time later)

```bash
time hf download MODEL_ID --exclude "*.pth" "original/*"
```

**YOU SHOULD SEE:** a progress bar, then `real  XmYs`. An 8B model is ~16 GB: about 2-3 minutes at 1 Gbit/s.

> **WATCH OUT:** some repos also ship `original/` (consolidated weights) or `.pth` copies. Excluding them saves half the download. Gated model → `403/401`: accept the licence on the model page (P01 Section 3), with the same account as the token.

## SECTION 5 — Load the model on the GPU and prove it generates

**WHERE:** GPU terminal
**GOAL:** load time, device map, parameter count, peak VRAM, and one generated sentence

Step 1. Paste the whole box and press Enter (replace `MODEL_ID` on the first line):

```bash
MID="MODEL_ID" python - <<'PY'
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = os.environ["MID"]; t0 = time.time()
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
print("load seconds:", round(time.time() - t0, 1))
print("device map:", set(getattr(model, "hf_device_map", {"": "single"}).values()))
print("params (B):", round(sum(p.numel() for p in model.parameters()) / 1e9, 3))
print("peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
enc = tok.apply_chat_template([{"role": "user", "content": "Say OK in one word."}], add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=16, do_sample=False)
print("answer:", tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

**YOU SHOULD SEE (shape, for an 8B model on one 80 GB GPU):**

```text
load seconds: 25.3
device map: {0}
params (B): 8.191
peak VRAM GB: 15.3
answer: OK
```

| Check | Healthy | If not |
|---|---|---|
| `peak VRAM GB` | ≈ the planner's weights figure (15.3 GB for 8B bf16) | ~30 GB = loaded in fp32 (forgot `dtype`) |
| `device map` | `{0}` (one GPU), or the GPUs you intended | contains `'cpu'` or `'disk'` = it did not fit and spilled: very slow. Use QLoRA or a bigger GPU |
| `answer` | coherent | gibberish = wrong tokenizer/template, or a base (non-instruct) model |
| an error about `BatchEncoding` | an old snippet without `return_dict=True` | use the box above exactly (transformers 5 returns a dict) |

Step 2. (QLoRA plan) check the 4-bit footprint:

```bash
MID="MODEL_ID" python - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained(os.environ["MID"], quantization_config=q, device_map={"": 0})
print("4-bit peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
PY
```

**YOU SHOULD SEE:** about 5.5-6 GB for an 8B model (embeddings and norms stay 16-bit). On MoE models expect much less saving (T03 §3).

---

## CHECKPOINT

- [ ] `gpu_preflight.sh` printed OK (or I chose a matching `TORCH_INDEX` and recorded why).
- [ ] `train env OK … cuda available: True`.
- [ ] `.env` loaded, `hf auth whoami` shows my user, `HF_HOME` is on the big disk.
- [ ] The model loads on the GPU, peak VRAM ≈ the plan, and it answers coherently.
