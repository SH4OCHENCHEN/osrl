#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/main.py}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/artifacts/rebuttal_runs}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/artifacts/logs}"

ALGO="${ALGO:-FLOWNFSW}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS=(${SEEDS_STR:-0})
TASKS=(${TASKS_STR:-OfflineCarGoal1 OfflineCarGoal2 OfflinePointGoal1 OfflinePointGoal2 OfflineCarButton1 OfflineCarButton2 OfflinePointButton1 OfflinePointButton2 OfflineCarPush1 OfflineCarPush2 OfflinePointPush1 OfflinePointPush2 OfflineCarRun OfflineAntRun OfflineDroneRun OfflineCarCircle OfflineDroneCircle OfflineAntCircle OfflineBallCircle OfflineBallRun})
CHUNKING_LENGTHS=(${CHUNKS_STR:-5})
CFG_GUIDANCES=(${CFG_GUIDANCES_STR:-})

MAX_TIMESTEP="${MAX_TIMESTEP:-}"
EVAL_FREQ="${EVAL_FREQ:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
TARGET_COST="${TARGET_COST:-5}"
EVAL_EPISODE="${EVAL_EPISODE:-20}"

TORCH_COMPILE="${TORCH_COMPILE:-auto}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"

SAVE_MODEL="${SAVE_MODEL:-0}"
USE_WANDB="${USE_WANDB:-0}"
FINAL_TEST="${FINAL_TEST:-0}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

common_args=(
  --algo "$ALGO"
  --device "$DEVICE"
  --output-root "$OUT_ROOT"
  --target-cost "$TARGET_COST"
  --eval-episode "$EVAL_EPISODE"
  --torch-compile "$TORCH_COMPILE"
  --compile-mode "$COMPILE_MODE"
  --matmul-precision "$MATMUL_PRECISION"
)

[[ -n "$MAX_TIMESTEP" ]] && common_args+=(--max-timestep "$MAX_TIMESTEP")
[[ -n "$EVAL_FREQ" ]] && common_args+=(--eval-freq "$EVAL_FREQ")
[[ -n "$BATCH_SIZE" ]] && common_args+=(--batch-size "$BATCH_SIZE")
[[ "$SAVE_MODEL" == "1" ]] && common_args+=(--save-model)
[[ "$USE_WANDB" == "1" ]] && common_args+=(--wandb)
[[ "$FINAL_TEST" == "1" ]] && common_args+=(--final-test)

run_one() {
  local task="$1"
  local seed="$2"
  local chunk="$3"
  local cfg="$4"
  local run_name="${ALGO}_${task}_seed${seed}"
  local extra_args=(--env_name "$task" --seed "$seed")

  if [[ -n "$chunk" ]]; then
    run_name="${run_name}_h${chunk}"
    extra_args+=(--chunking-length "$chunk")
  fi

  if [[ -n "$cfg" ]]; then
    run_name="${run_name}_cfg${cfg}"
    extra_args+=(--cfg-guidance "$cfg")
  fi

  local run_log="${LOG_DIR}/${run_name}.log"
  echo "[$(date '+%F %T')] START $run_name"
  echo "$PYTHON_BIN $TRAIN_SCRIPT ${common_args[*]} ${extra_args[*]} --run-name $run_name"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" "${common_args[@]}" "${extra_args[@]}" --run-name "$run_name" 2>&1 | tee "$run_log"
  echo "[$(date '+%F %T')] END $run_name"
}

echo "Project root : $PROJECT_ROOT"
echo "Train script : $TRAIN_SCRIPT"
echo "Output root  : $OUT_ROOT"
echo "Log dir      : $LOG_DIR"
echo "Algorithm    : $ALGO"
echo "Tasks        : ${TASKS[*]}"
echo "Seeds        : ${SEEDS[*]}"
echo "Chunks       : ${CHUNKING_LENGTHS[*]}"
echo "CFG values   : ${CFG_GUIDANCES[*]:-<none>}"
echo

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "${#CFG_GUIDANCES[@]}" -gt 0 ]]; then
      for cfg in "${CFG_GUIDANCES[@]}"; do
        for chunk in "${CHUNKING_LENGTHS[@]}"; do
          run_one "$task" "$seed" "$chunk" "$cfg"
        done
      done
    else
      for chunk in "${CHUNKING_LENGTHS[@]}"; do
        run_one "$task" "$seed" "$chunk" ""
      done
    fi
  done
done

echo "All tasks finished at $(date '+%F %T')"
