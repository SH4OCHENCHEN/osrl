#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CHUNKS_STR="${CHUNKS_STR:-1}"
export OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/artifacts/action_chunk_len1_runs}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/artifacts/action_chunk_len1_logs}"
export USE_WANDB="${USE_WANDB:-0}"
export SAVE_MODEL="${SAVE_MODEL:-0}"
export FINAL_TEST="${FINAL_TEST:-0}"

if [[ "$FINAL_TEST" == "1" && "$SAVE_MODEL" != "1" ]]; then
  echo "FINAL_TEST=1 requires a saved best model; enabling SAVE_MODEL=1."
  export SAVE_MODEL=1
fi

echo "Running action chunking length = ${CHUNKS_STR} on all configured tasks."
echo "Results: ${OUT_ROOT}"
echo "Logs   : ${LOG_DIR}"
echo

exec bash "${SCRIPT_DIR}/train_all_tasks.sh"
