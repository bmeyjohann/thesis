#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/benjamin/miniconda3/envs/fasttd3/bin/python" ]]; then
    PYTHON_BIN="/home/benjamin/miniconda3/envs/fasttd3/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No usable python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 127
  fi
fi

WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-maze-pixels}"
ENV_NAME="${ENV_NAME:-pointmaze-arena-danger-lethal-v0}"
MPLCONFIGDIR_VALUE="${MPLCONFIGDIR:-/tmp/matplotlib}"
WANDB_DIR_VALUE="${WANDB_DIR:-${PWD}/wandb}"
WANDB_CACHE_DIR_VALUE="${WANDB_CACHE_DIR:-${PWD}/.cache/wandb}"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-50000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ACTION_REPEAT="${ACTION_REPEAT:-2}"
FRAME_STACK="${FRAME_STACK:-3}"
PIXEL_WIDTH="${PIXEL_WIDTH:-84}"
PIXEL_HEIGHT="${PIXEL_HEIGHT:-84}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED="${SEED:-1}"
VIZ_ON_CHECKPOINT="${VIZ_ON_CHECKPOINT:-1}"
VIZ_GRID_RESOLUTION="${VIZ_GRID_RESOLUTION:-32}"
VIZ_QUIVER_STRIDE="${VIZ_QUIVER_STRIDE:-2}"
TEACHER_TYPE="${TEACHER_TYPE:-bfs}"
INTERVENTION_MODE="${INTERVENTION_MODE:-agent}"
TOLERANCE_TYPE="${TOLERANCE_TYPE:-angle}"
TOLERANCE_VALUE="${TOLERANCE_VALUE:-30.0}"
HARD_BLOCK_LETHAL="${HARD_BLOCK_LETHAL:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
ENV_TAG="${ENV_NAME//-/_}"
EXP_NAME="${EXP_NAME:-maze_pixels_own_${ENV_TAG}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs
mkdir -p "${MPLCONFIGDIR_VALUE}" "${WANDB_DIR_VALUE}" "${WANDB_CACHE_DIR_VALUE}"

ARGS=(
  --env_name "${ENV_NAME}"
  --obs_mode pixels
  --pixel_width "${PIXEL_WIDTH}"
  --pixel_height "${PIXEL_HEIGHT}"
  --device auto
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --save_interval "${SAVE_INTERVAL}"
  --eval_every_frames "${EVAL_EVERY_FRAMES}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --log_interval "${LOG_INTERVAL}"
  --batch_size "${BATCH_SIZE}"
  --action_repeat "${ACTION_REPEAT}"
  --frame_stack "${FRAME_STACK}"
  --learning_rate "${LEARNING_RATE}"
  --reward_type dense
  --dense_reward_scale 1.0
  --use_intervention
  --intervention_mode "${INTERVENTION_MODE}"
  --teacher_type "${TEACHER_TYPE}"
  --tolerance_type "${TOLERANCE_TYPE}"
  --tolerance_value "${TOLERANCE_VALUE}"
  --seed "${SEED}"
  --project "${PROJECT}"
  --exp_name "${EXP_NAME}"
)

if [[ "${HARD_BLOCK_LETHAL}" == "1" ]]; then
  ARGS+=(--hard_block_lethal)
fi
if [[ "${VIZ_ON_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--viz_on_checkpoint --viz_grid_resolution "${VIZ_GRID_RESOLUTION}" --viz_quiver_stride "${VIZ_QUIVER_STRIDE}")
fi

echo "=== Starting ${EXP_NAME} (maze pixels own) ==="
echo "Log file: ${LOG_FILE}"

MPLCONFIGDIR="${MPLCONFIGDIR_VALUE}" \
WANDB_DIR="${WANDB_DIR_VALUE}" \
WANDB_CACHE_DIR="${WANDB_CACHE_DIR_VALUE}" \
WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_drqv2_ogbench.py \
  "${ARGS[@]}" \
  --use_wandb \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
