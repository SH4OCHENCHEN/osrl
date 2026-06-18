#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_DATASET_REPO="${HF_DATASET_REPO:-YYY-45/DSRL}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATASET_DIR="${DATASET_DIR:-${HOME}/.dsrl/datasets}"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$PYTHON_BIN" "${SCRIPT_DIR}/download_dsrl_datasets.py" \
  --repo-id "$HF_DATASET_REPO" \
  --endpoint "$HF_ENDPOINT" \
  --dataset-dir "$DATASET_DIR" \
  "$@"
