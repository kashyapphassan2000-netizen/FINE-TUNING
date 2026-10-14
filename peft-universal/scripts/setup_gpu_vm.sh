#!/usr/bin/env bash
# GPU VM (Ubuntu): system packages + uv + two isolated virtual environments.
#   .venv-train : torch 2.14 + transformers/peft/trl/bitsandbytes   (training, merging)
#   .venv-serve : vllm 0.29.0 (which pins torch==2.13.0)             (serving)
# They cannot share one env: vLLM pins an older torch than the training stack.
# Works on a VM with sudo (HyperStack: user ubuntu) AND in root containers without sudo (RunPod, Vast.ai).
# Older driver (CUDA < 13)? Install a torch build for your CUDA first, e.g.:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_gpu_vm.sh
# (check which CUDA variants exist for torch 2.14 at https://pytorch.org/get-started/locally/ first)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
$SUDO apt-get update -y
$SUDO apt-get install -y git git-lfs tmux htop nvtop build-essential python3-dev rclone || \
  $SUDO apt-get install -y git git-lfs tmux htop build-essential python3-dev   # some images lack nvtop/rclone
git lfs install
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv .venv-train --python 3.11
if [ -n "${TORCH_INDEX:-}" ]; then
  # torch from the index matching YOUR driver; requirements' torch==2.14.0 is then already satisfied
  uv pip install --python .venv-train/bin/python "torch==2.14.0" --index-url "$TORCH_INDEX"
fi
uv pip install --python .venv-train/bin/python -r requirements-train.txt
uv pip install --python .venv-train/bin/python -e .
uv venv .venv-serve --python 3.11
uv pip install --python .venv-serve/bin/python -r requirements-serve.txt
.venv-train/bin/python - <<'PY'
import torch, peft, transformers, trl, bitsandbytes
print("train env OK | torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count(), "| bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("peft", peft.__version__, "transformers", transformers.__version__, "trl", trl.__version__, "bnb", bitsandbytes.__version__)
PY
.venv-serve/bin/python -c "import vllm, torch; print('serve env OK | vllm', vllm.__version__, '| torch', torch.__version__)"
