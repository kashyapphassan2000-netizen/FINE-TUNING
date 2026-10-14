# 24. Hyperstack GPU VM: deploy, connect, preflight, install

## 24.1 SSH key on the laptop

```laptop
ssh-keygen -t ed25519 -f $HOME\.ssh\hyperstack -C "hyperstack"
Get-Content $HOME\.ssh\hyperstack.pub      # paste this into Hyperstack (chapter 21.2 step 5)
```

## 24.2 Deploy the VM (console)

```browser
1. Virtual Machines → Deploy New Virtual Machine.
2. Flavor: the GPU chosen in chapter 23 (the docs' quickstart example is n3-L40x1: 1x L40 48 GB, 28 vCPU, 58 GB RAM).
3. Environment: yours.
4. OS image: pick an Ubuntu image whose NVIDIA DRIVER supports CUDA 13 (see warning). Image names include the
   driver and CUDA version, e.g. "Server 24.04 LTS R570 CUDA 12.8"; a "with Docker" variant includes Docker,
   Compose and the NVIDIA Container Toolkit.
5. SSH key: yours. Enable SSH access (port 22). Assign a public IP.
6. Deploy. Wait until status is ACTIVE and a public IP is shown (billing starts at ACTIVE).
```

:::warn The biggest trap in this guide
Hyperstack's default image (per its getting-started docs) is "Ubuntu Server 22.04 LTS R535 CUDA 12.2". The PyPI wheels of torch 2.14.0 (training) and 2.13.0 (pulled by vLLM 0.29.0) are built for CUDA 13.0, which needs a newer driver than R535/R570. On an old driver, `torch.cuda.is_available()` is False and training silently runs on CPU or fails. `scripts/gpu_preflight.sh` stops you. Fixes, in order: choose an image with a driver that reports CUDA ≥ 13.0 in `nvidia-smi`; or run training/serving in an NVIDIA container that matches your driver; or install torch builds for your CUDA version from download.pytorch.org (then vLLM must match too). Hyperstack's docs say changing drivers yourself is outside their support.
:::

## 24.3 Connect: SSH, VS Code Remote-SSH, WinSCP

```text FILE  C:\Users\<you>\.ssh\config
Host hs-train
    HostName <PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/hyperstack
    ServerAliveInterval 60
```

```laptop
ssh hs-train          # first time: type yes
```

VS Code: `F1` → **Remote-SSH: Connect to Host** → `hs-train` → **Open Folder** `/home/ubuntu`. Terminals opened now run on the GPU VM.

WinSCP: New Session → SFTP → host = public IP, user = ubuntu → Advanced → SSH → Authentication → private key file (WinSCP converts it to .ppk when asked) → Login. Use it to upload `data/processed/*.jsonl` or your document folder.

## 24.4 Preflight (before installing anything)

@file scripts/gpu_preflight.sh

```gpu
git clone git@github.com:<you>/peft-universal.git && cd peft-universal   # or upload the ZIP with WinSCP
bash scripts/gpu_preflight.sh
```

```out Healthy example shape
== GPU
name, memory.total [MiB], driver_version
<GPU NAME>, <MEM> MiB, <DRIVER>
driver major=<N>  max CUDA supported by driver=13.x
OK: driver can run CUDA 13 wheels
```

## 24.5 Install (two environments)

@file scripts/setup_gpu_vm.sh

|Installed|One-line why|
|---|---|
|git, git-lfs|clone code; large files on the Hub|
|tmux|training survives a dropped SSH connection|
|htop, nvtop|watch CPU and GPU usage|
|build-essential, python3-dev|compile any package without a wheel|
|rclone|Google Drive sync|
|uv + Python 3.11|fast, reproducible environments|
|torch 2.14 (CUDA 13)|tensor math on the GPU|
|transformers|model architectures and loading|
|peft|LoRA/QLoRA/DoRA/AdaLoRA/LoRA+ layers|
|trl|SFTTrainer: chat templates, masking, packing|
|accelerate|multi-GPU launching (DDP/FSDP)|
|bitsandbytes|4-bit NF4 and 8-bit/paged optimizers|
|datasets|streaming jsonl loading|
|vllm 0.29.0 (own env)|fast OpenAI-compatible serving|

The NVIDIA driver and CUDA runtime come with the image and the torch wheels (they bundle CUDA libraries); you do not install the CUDA toolkit for this template.

```gpu
bash scripts/setup_gpu_vm.sh
cp .env.example .env && nano .env        # HF_TOKEN etc.
export HF_HOME=/ephemeral/hf-cache       # if the VM has an ephemeral NVMe mount; check with df -h
```

```out Shape of the success lines
train env OK | torch 2.14.0+cu130 | cuda available: True | gpus: 1 | bf16: True
peft 0.21.0 transformers 5.17.0 trl 1.13.0 bnb 0.50.2
serve env OK | vllm 0.29.0 | torch 2.13.0+cu130
```

:::check
- preflight printed OK.
- `cuda available: True` in the train env.
- `.env` filled on the VM; `hf auth whoami` (inside `.venv-train`) shows my user.
:::
