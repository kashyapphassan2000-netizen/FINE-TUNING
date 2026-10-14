#!/usr/bin/env bash
# GPU VM: check driver/CUDA/disk/RAM BEFORE installing anything. Exit 1 if the box cannot run torch 2.13/2.14.
set -euo pipefail
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "0.0")
echo "driver major=$DRV  max CUDA supported by driver=$CUDA"
echo "== CPU/RAM/disk"; nproc; free -g | head -2; df -h / | tail -1
echo "== NVLink / topology"; nvidia-smi topo -m || true
if [ "${CUDA%%.*}" -lt 13 ]; then
  echo "WARNING: driver supports CUDA $CUDA. PyPI torch 2.13/2.14 wheels are built for CUDA 13.0."
  echo "Fix (pick one): (1) redeploy with a newer image / host (driver R580+), or"
  echo "  (2) install a torch build for YOUR CUDA: TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_gpu_vm.sh"
  echo "      (check the CUDA variants of torch 2.14 on https://pytorch.org/get-started/locally/ first), or"
  echo "  (3) run training/vLLM inside the matching NVIDIA/vLLM container."
  if [ "${ALLOW_OLD_CUDA:-0}" = "1" ]; then echo "ALLOW_OLD_CUDA=1: continuing"; exit 0; fi
  exit 1
fi
echo "OK: driver can run CUDA 13 wheels"
