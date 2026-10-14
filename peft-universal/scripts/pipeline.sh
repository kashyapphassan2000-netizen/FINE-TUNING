#!/usr/bin/env bash
# End-to-end on the GPU VM:  METHOD=qlora MODEL_CFG=configs/models/llama-3.1-8b.yaml bash scripts/pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
METHOD=${METHOD:-lora}
MODEL_CFG=${MODEL_CFG:-configs/models/smollm2-135m.yaml}
PY=.venv-train/bin/python
OUT=$($PY - <<PY
from peftkit.config import load_config
print(load_config("configs/base.yaml","configs/methods/${METHOD}.yaml","${MODEL_CFG}").train.output_dir)
PY
)
# SKIP_SYNTH=1: you already built data/processed/train.jsonl (e.g. on the laptop) and uploaded it.
# Then the data stage is skipped entirely, so the uploaded files are never overwritten.
if [ "${SKIP_SYNTH:-0}" = "1" ]; then
  [ -s data/processed/train.jsonl ] || { echo "SKIP_SYNTH=1 but data/processed/train.jsonl is missing or empty"; exit 1; }
  echo "SKIP_SYNTH=1: using existing data/processed/train.jsonl ($(wc -l < data/processed/train.jsonl) rows)"
else
  $PY -m peftkit.data.ingest --src data/raw --out data/interim/chunks.jsonl
  $PY -m peftkit.data.synth --chunks data/interim/chunks.jsonl --out data/interim/synthetic.jsonl --pairs-per-chunk 3
  $PY -m peftkit.data.format --inputs data/interim/synthetic.jsonl ${EXTRA_DATA:-} --out-dir data/processed
fi
$PY -m peftkit.train --config configs/base.yaml "configs/methods/${METHOD}.yaml" "$MODEL_CFG" ${TRAIN_SET:+--set $TRAIN_SET}
$PY -m peftkit.serve_check "$OUT/adapter" || true
$PY -m peftkit.merge --adapter "$OUT/adapter" --out "$OUT/merged"
echo "done: adapter=$OUT/adapter merged=$OUT/merged"
