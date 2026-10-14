#!/usr/bin/env bash
# Public benchmarks with lm-evaluation-harness (lm-eval 0.4.13; flags read from its source).
# Install in its own env:  uv venv .venv-eval && uv pip install --python .venv-eval/bin/python "lm-eval[vllm]==0.4.13"
#
# FACT read from lm-eval source: the chat-completions backend raises
#   "Loglikelihood is not supported for chat completions" -> MMLU / HellaSwag / ARC / Winogrande
#   (multiple-choice, loglikelihood) CANNOT run through a chat API. Run them with the vllm or hf backend.
set -euo pipefail
MODEL=${MODEL:?set MODEL to a Hub id or local merged dir}
BASE=${BASE:-}            # optional: the base model, to report before/after on the same tasks
OUT=${OUT:-reports/benchmarks}
mkdir -p "$OUT"
run_local () {   # $1 = model path/id, $2 = label
  lm_eval --model vllm --model_args "pretrained=$1,dtype=bfloat16,gpu_memory_utilization=0.85" \
    --tasks mmlu,hellaswag,arc_challenge,gsm8k,ifeval --batch_size auto --apply_chat_template \
    --output_path "$OUT/$2" --log_samples
}
[ -n "$BASE" ] && run_local "$BASE" base
run_local "$MODEL" tuned
# Against the DEPLOYED endpoint (generation tasks only). OPENAI_API_KEY = a gateway key.
if [ -n "${ENDPOINT:-}" ]; then
  OPENAI_API_KEY=${API_KEY:?set API_KEY} lm_eval --model local-chat-completions \
    --model_args "model=${SERVED_NAME:-mymodel},base_url=$ENDPOINT/chat/completions,num_concurrent=4,max_retries=3" \
    --tasks gsm8k,ifeval --apply_chat_template --output_path "$OUT/served"
fi
echo "results in $OUT (compare base vs tuned per task; a general-benchmark drop = forgetting)"

# ---- ADDED: benchmark the ADAPTER without merging (hf backend accepts peft=, read from lm-eval 0.4.13 source),
#      then gate on forgetting. Use when you want the check BEFORE the merge step.
if [ -n "${ADAPTER:-}" ]; then
  lm_eval --model hf --model_args "pretrained=${BASE:?set BASE},peft=$ADAPTER,dtype=bfloat16" \
    --tasks mmlu,gsm8k --batch_size auto --apply_chat_template --output_path "$OUT/tuned_adapter"
fi
if [ -n "$BASE" ] && [ -d "$OUT/base" ]; then
  python -m peftkit.bench_gate --base "$OUT/base" --tuned "${TUNED_DIR:-$OUT/tuned}" --max-drop "${MAX_DROP:-0.02}" --out reports/bench_gate.json
fi
