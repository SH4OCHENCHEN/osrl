#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-rebuttal_supp}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/main.py}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/artifacts/rebuttal_runs}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/artifacts/tmux_logs}"
SAVE_MODEL="${SAVE_MODEL:-0}"
USE_WANDB="${USE_WANDB:-0}"
FINAL_TEST="${FINAL_TEST:-0}"

if [[ "$USE_WANDB" != "1" ]]; then
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

ENVS_DEFAULT=(OfflineCarGoal1 OfflineCarGoal2)
SEEDS_DEFAULT=(0)
OMEGAS_DEFAULT=(1 2 3 4)
CHUNKS_DEFAULT=(1 5 10 15)

if [[ "${1:-}" != "--inside-tmux" ]]; then
  mkdir -p "$LOG_DIR" "$OUT_ROOT"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
  fi

  export SESSION_NAME PYTHON_BIN SCRIPT_DIR PROJECT_ROOT TRAIN_SCRIPT DEVICE OUT_ROOT LOG_DIR SAVE_MODEL USE_WANDB FINAL_TEST
  export ENVS_STR="${ENVS_STR:-${ENVS_DEFAULT[*]}}"
  export SEEDS_STR="${SEEDS_STR:-${SEEDS_DEFAULT[*]}}"
  export OMEGAS_STR="${OMEGAS_STR:-${OMEGAS_DEFAULT[*]}}"
  export CHUNKS_STR="${CHUNKS_STR:-${CHUNKS_DEFAULT[*]}}"
  export MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/tmux_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log}"

  tmux new-session -d -s "$SESSION_NAME" "bash '$0' --inside-tmux"
  echo "Launched tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Master log: $MASTER_LOG"
  exit 0
fi

if [[ -n "${ENVS_STR:-}" ]]; then
  IFS=' ' read -r -a ENVS <<< "$ENVS_STR"
else
  ENVS=("${ENVS_DEFAULT[@]}")
fi

if [[ -n "${SEEDS_STR:-}" ]]; then
  IFS=' ' read -r -a SEEDS <<< "$SEEDS_STR"
else
  SEEDS=("${SEEDS_DEFAULT[@]}")
fi

if [[ -n "${OMEGAS_STR:-}" ]]; then
  IFS=' ' read -r -a OMEGAS <<< "$OMEGAS_STR"
else
  OMEGAS=("${OMEGAS_DEFAULT[@]}")
fi

if [[ -n "${CHUNKS_STR:-}" ]]; then
  IFS=' ' read -r -a CHUNKS <<< "$CHUNKS_STR"
else
  CHUNKS=("${CHUNKS_DEFAULT[@]}")
fi

mkdir -p "$LOG_DIR" "$OUT_ROOT"
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/tmux_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$MASTER_LOG") 2>&1

run_one() {
  local run_name="$1"
  shift
  local run_log="${LOG_DIR}/${run_name}.log"

  echo "============================================================"
  echo "[$(date '+%F %T')] START ${run_name}"
  echo "Command: $PYTHON_BIN $TRAIN_SCRIPT $*"
  echo "Run log: $run_log"
  echo "============================================================"

  "$PYTHON_BIN" "$TRAIN_SCRIPT" "$@" 2>&1 | tee "$run_log"

  echo "[$(date '+%F %T')] END ${run_name}"
  echo
}

common_args=(--device "$DEVICE" --output-root "$OUT_ROOT")
if [[ "$SAVE_MODEL" == "1" ]]; then
  common_args+=(--save-model)
fi
if [[ "$USE_WANDB" == "1" ]]; then
  common_args+=(--wandb)
fi
if [[ "$FINAL_TEST" == "1" ]]; then
  common_args+=(--final-test)
fi

echo "Session      : $SESSION_NAME"
echo "Train script : $TRAIN_SCRIPT"
echo "Device       : $DEVICE"
echo "Output root  : $OUT_ROOT"
echo "Log dir      : $LOG_DIR"
echo "Master log   : $MASTER_LOG"
echo "Envs         : ${ENVS[*]}"
echo "Seeds        : ${SEEDS[*]}"
echo "Omegas       : ${OMEGAS[*]}"
echo "Chunks       : ${CHUNKS[*]}"
echo

# for env in "${ENVS[@]}"; do
#   for seed in "${SEEDS[@]}"; do
#     for omega in "${OMEGAS[@]}"; do
#       run_name="FLOWNFS_${env}_seed${seed}_cfg${omega}"
#       run_one "$run_name" \
#         --algo FLOWNFS \
#         --env_name "$env" \
#         --seed "$seed" \
#         --run-name "$run_name" \
#         --cfg-guidance "$omega" \
#         "${common_args[@]}"
#     done
#   done
# done

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for h in "${CHUNKS[@]}"; do
      run_name="FLOWNFGAU_${env}_seed${seed}_h${h}"
      run_one "$run_name" \
        --algo FLOWNFSW \
        --env_name "$env" \
        --seed "$seed" \
        --run-name "$run_name" \
        --chunking-length "$h" \
        "${common_args[@]}"
    done
  done
done

echo "All rebuttal experiments finished at $(date '+%F %T')"
