#!/usr/bin/env bash
# Two modes:
#   merged : MODEL=user/merged-model bash scripts/serve_vllm.sh
#   adapter: MODEL=base/model ADAPTERS="support=user/adapter-a legal=user/adapter-b" MAX_LORA_RANK=16 bash scripts/serve_vllm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${MODEL:?set MODEL}"
: "${VLLM_API_KEY:?set VLLM_API_KEY in .env}"
TP=${TP:-$(nvidia-smi -L | wc -l)}
ARGS=(serve "$MODEL"
  --host 0.0.0.0 --port "${PORT:-8000}"
  --served-model-name "${SERVED_NAME:-$MODEL}"
  --tensor-parallel-size "$TP"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${GPU_UTIL:-0.90}"
  --api-key "$VLLM_API_KEY")
if [ -n "${ADAPTERS:-}" ]; then
  ARGS+=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --max-loras "${MAX_LORAS:-4}" --lora-modules)
  for a in $ADAPTERS; do ARGS+=("$a"); done
fi
exec .venv-serve/bin/vllm "${ARGS[@]}"
