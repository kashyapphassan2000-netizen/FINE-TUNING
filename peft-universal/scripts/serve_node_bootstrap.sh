#!/usr/bin/env bash
# Runs ON a fresh serving VM (piped over SSH by serving/scaler.py). Needs REPO_URL and secrets in the env
# or a pre-baked snapshot. Keeps it minimal: serve env only, merged model from the Hub.
set -euo pipefail
: "${REPO_URL:=https://github.com/YOUR_USER/peft-universal.git}"
[ -d peft-universal ] || git clone "$REPO_URL" peft-universal
cd peft-universal
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-serve --python 3.11 && uv pip install --python .venv-serve/bin/python -r requirements-serve.txt
tmux new-session -d -s vllm "bash scripts/serve_vllm.sh 2>&1 | tee vllm.log"
for i in $(seq 1 120); do curl -sf localhost:8000/health && exit 0; sleep 10; done
echo "vLLM did not become healthy"; exit 1
