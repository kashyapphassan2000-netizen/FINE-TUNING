#!/usr/bin/env bash
# End-to-end domain SFT (+ optional DPO) dataset build. Laptop or VM; only needs an OpenAI-compatible API.
#   DOMAIN="industrial pump support" bash scripts/build_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
PY=${PY:-python}
DOMAIN=${DOMAIN:?set DOMAIN}
mkdir -p data/interim data/processed reports
# 1) grounded data from your documents
$PY -m peftkit.data.ingest --src data/raw --out data/interim/chunks.jsonl
$PY -m peftkit.data.synth --chunks data/interim/chunks.jsonl --out data/interim/grounded.jsonl --pairs-per-chunk 3
# 2) diverse task data from ~20+ human seed tasks
if [ -f data/seeds.jsonl ]; then
  $PY -m peftkit.data.self_instruct --seeds data/seeds.jsonl --domain "$DOMAIN" --n "${SELF_INSTRUCT_N:-300}" \
      --out data/interim/self_instruct.jsonl
  $PY -m peftkit.data.format --inputs data/interim/self_instruct.jsonl --out-dir data/interim/si_fmt --eval-ratio 0
  SI=data/interim/si_fmt/train.jsonl
fi
$PY -m peftkit.data.format --inputs data/interim/grounded.jsonl ${SI:-} --out-dir data/interim/fmt --eval-ratio 0
# 3) dedup + filters (+ contamination guard against your held-out eval set, if you have one)
$PY -m peftkit.data.quality --inputs data/interim/fmt/train.jsonl --out data/interim/clean.jsonl \
    --report reports/quality.json ${EVAL_GUARD:+--contamination $EVAL_GUARD}
# 4) LLM-as-judge filter (use a different model than the generator if you can)
$PY -m peftkit.data.judge --input data/interim/clean.jsonl --out data/interim/judged.jsonl --min-score 4 \
    ${JUDGE_MODEL:+--judge-model $JUDGE_MODEL}
# 5) final split
$PY -m peftkit.data.format --inputs data/interim/judged.jsonl --out-dir data/processed --eval-ratio 0.05 \
    ${TOKENIZER:+--tokenizer $TOKENIZER}
# 6) optional preference pairs for DPO
if [ "${MAKE_DPO:-0}" = "1" ]; then
  $PY -m peftkit.data.preference --input data/processed/train.jsonl --out data/processed/preference.jsonl \
      --k 4 --min-margin 2 ${JUDGE_MODEL:+--judge-model $JUDGE_MODEL} ${WEAK_MODEL:+--weak-model $WEAK_MODEL}
fi
echo "dataset ready in data/processed"
