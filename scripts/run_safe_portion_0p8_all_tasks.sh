#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/main.py}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/artifacts/safe_portion_0p8_runs}"

ALGO="${ALGO:-FLOWNFS}"
DEVICE="${DEVICE:-cuda:0}"
SAFE_PORTION="${SAFE_PORTION:-0.8}"
SEEDS=(${SEEDS_STR:-0})
TASKS=(${TASKS_STR:-OfflineCarGoal1 OfflineCarGoal2 OfflinePointGoal1 OfflinePointGoal2 OfflineCarButton1 OfflineCarButton2 OfflinePointButton1 OfflinePointButton2 OfflineCarPush1 OfflineCarPush2 OfflinePointPush1 OfflinePointPush2 OfflineCarRun OfflineAntRun OfflineDroneRun OfflineCarCircle OfflineDroneCircle OfflineAntCircle OfflineBallCircle OfflineBallRun})

CHUNKING_LENGTHS=(${CHUNKS_STR:-})
CFG_GUIDANCES=(${CFG_GUIDANCES_STR:-})

MAX_TIMESTEP="${MAX_TIMESTEP:-}"
EVAL_FREQ="${EVAL_FREQ:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
TARGET_COST="${TARGET_COST:-}"
EVAL_EPISODE="${EVAL_EPISODE:-20}"
TEST_EPISODES="${TEST_EPISODES:-60}"
DATASET_DOWNLOAD="${DATASET_DOWNLOAD:-auto}"
HF_DATASET_REPO="${HF_DATASET_REPO:-YYY-45/DSRL}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATASET_DIR="${DATASET_DIR:-}"

TORCH_COMPILE="${TORCH_COMPILE:-auto}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"

SAVE_MODEL="${SAVE_MODEL:-1}"
USE_WANDB="${USE_WANDB:-0}"
FINAL_TEST="${FINAL_TEST:-1}"

if [[ "$USE_WANDB" != "1" ]]; then
  export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

if [[ "$FINAL_TEST" == "1" && "$SAVE_MODEL" != "1" ]]; then
  echo "FINAL_TEST=1 requires a saved best model; enabling SAVE_MODEL=1."
  SAVE_MODEL=1
fi

mkdir -p "$OUT_ROOT"

common_args=(
  --algo "$ALGO"
  --device "$DEVICE"
  --output-root "$OUT_ROOT"
  --safe-portion "$SAFE_PORTION"
  --eval-episode "$EVAL_EPISODE"
  --test-episodes "$TEST_EPISODES"
  --dataset-download "$DATASET_DOWNLOAD"
  --hf-dataset-repo "$HF_DATASET_REPO"
  --hf-endpoint "$HF_ENDPOINT"
  --torch-compile "$TORCH_COMPILE"
  --compile-mode "$COMPILE_MODE"
  --matmul-precision "$MATMUL_PRECISION"
)

[[ -n "$MAX_TIMESTEP" ]] && common_args+=(--max-timestep "$MAX_TIMESTEP")
[[ -n "$EVAL_FREQ" ]] && common_args+=(--eval-freq "$EVAL_FREQ")
[[ -n "$BATCH_SIZE" ]] && common_args+=(--batch-size "$BATCH_SIZE")
[[ -n "$TARGET_COST" ]] && common_args+=(--target-cost "$TARGET_COST")
[[ -n "$DATASET_DIR" ]] && common_args+=(--dataset-dir "$DATASET_DIR")
[[ "$SAVE_MODEL" == "1" ]] && common_args+=(--save-model)
[[ "$USE_WANDB" == "1" ]] && common_args+=(--wandb)
[[ "$FINAL_TEST" == "1" ]] && common_args+=(--final-test)

run_one() {
  local task="$1"
  local seed="$2"
  local chunk="$3"
  local cfg="$4"
  local safe_tag="${SAFE_PORTION/./p}"
  local run_name="${ALGO}_${task}_seed${seed}_safe${safe_tag}"
  local extra_args=(--env_name "$task" --seed "$seed")

  if [[ -n "$chunk" ]]; then
    run_name="${run_name}_h${chunk}"
    extra_args+=(--chunking-length "$chunk")
  fi

  if [[ -n "$cfg" ]]; then
    run_name="${run_name}_cfg${cfg}"
    extra_args+=(--cfg-guidance "$cfg")
  fi

  echo "[$(date '+%F %T')] START $run_name"
  echo "$PYTHON_BIN $TRAIN_SCRIPT ${common_args[*]} ${extra_args[*]} --run-name $run_name"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" "${common_args[@]}" "${extra_args[@]}" --run-name "$run_name"
  echo "[$(date '+%F %T')] END $run_name"
  echo
}

echo "Project root : $PROJECT_ROOT"
echo "Train script : $TRAIN_SCRIPT"
echo "Output root  : $OUT_ROOT"
echo "Algorithm    : $ALGO"
echo "Device       : $DEVICE"
echo "Safe portion : $SAFE_PORTION"
echo "Tasks        : ${TASKS[*]}"
echo "Seeds        : ${SEEDS[*]}"
echo "Chunks       : ${CHUNKING_LENGTHS[*]:-<config default>}"
echo "CFG values   : ${CFG_GUIDANCES[*]:-<config default>}"
echo "Final test   : $FINAL_TEST (${TEST_EPISODES} episodes)"
echo "Dataset mode : $DATASET_DOWNLOAD"
echo "HF repo      : $HF_DATASET_REPO"
echo "HF endpoint  : ${HF_ENDPOINT:-<official>}"
echo
echo "Files will be organized as:"
echo "  $OUT_ROOT/models"
echo "  $OUT_ROOT/configs"
echo "  $OUT_ROOT/train_csv"
echo "  $OUT_ROOT/test_csv"
echo "  $OUT_ROOT/curves"
echo

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "${#CFG_GUIDANCES[@]}" -gt 0 ]]; then
      for cfg in "${CFG_GUIDANCES[@]}"; do
        if [[ "${#CHUNKING_LENGTHS[@]}" -gt 0 ]]; then
          for chunk in "${CHUNKING_LENGTHS[@]}"; do
            run_one "$task" "$seed" "$chunk" "$cfg"
          done
        else
          run_one "$task" "$seed" "" "$cfg"
        fi
      done
    elif [[ "${#CHUNKING_LENGTHS[@]}" -gt 0 ]]; then
      for chunk in "${CHUNKING_LENGTHS[@]}"; do
        run_one "$task" "$seed" "$chunk" ""
      done
    else
      run_one "$task" "$seed" "" ""
    fi
  done
done

echo "All safe_portion=${SAFE_PORTION} tasks finished at $(date '+%F %T')"
