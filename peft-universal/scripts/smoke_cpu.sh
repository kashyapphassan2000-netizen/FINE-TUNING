#!/usr/bin/env bash
# Laptop/CPU: full test suite (tiny random model, fake LLM API, fake S3, fake vLLM).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
