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
PROJECT="${PROJECT:-ogbench-maze-state}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-}"
ENV_NAME="${ENV_NAME:-pointmaze-arena-danger-lethal-v0}"
MPLCONFIGDIR_VALUE="${MPLCONFIGDIR:-/tmp/matplotlib}"
WANDB_DIR_VALUE="${WANDB_DIR:-${PWD}/wandb}"
WANDB_CACHE_DIR_VALUE="${WANDB_CACHE_DIR:-${PWD}/.cache/wandb}"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

NUM_ENVS="${NUM_ENVS:-16}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
SEED="${SEED:-42}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
USE_LAYER_NORM="${USE_LAYER_NORM:-1}"
HG_ENSEMBLE_SIZE="${HG_ENSEMBLE_SIZE:-5}"
HG_DOUBT_PERCENTILE="${HG_DOUBT_PERCENTILE:-75.0}"
DEMO_BUFFER_CAPACITY="${DEMO_BUFFER_CAPACITY:-200000}"
DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
VIZ_GRID_RESOLUTION="${VIZ_GRID_RESOLUTION:-32}"
VIZ_QUIVER_STRIDE="${VIZ_QUIVER_STRIDE:-2}"
EVAL_SAVE_TRAJECTORY_PLOT="${EVAL_SAVE_TRAJECTORY_PLOT:-1}"
EVAL_TRAJECTORY_PLOT_EPISODES="${EVAL_TRAJECTORY_PLOT_EPISODES:-5}"
VIZ_ON_CHECKPOINT="${VIZ_ON_CHECKPOINT:-1}"
TEACHER_TYPE="${TEACHER_TYPE:-bfs}"
INTERVENTION_MODE="${INTERVENTION_MODE:-agent}"
TOLERANCE_TYPE="${TOLERANCE_TYPE:-angle}"
TOLERANCE_VALUE="${TOLERANCE_VALUE:-30.0}"
HARD_BLOCK_LETHAL="${HARD_BLOCK_LETHAL:-1}"
INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-1.0}"
INTERVENTION_EPISODE_PROB_MIN="${INTERVENTION_EPISODE_PROB_MIN:-1.0}"
INTERVENTION_EPISODE_PROB_DECAY_STEPS="${INTERVENTION_EPISODE_PROB_DECAY_STEPS:-0}"
REWARD_TYPE="${REWARD_TYPE:-dense}"
DENSE_REWARD_SCALE="${DENSE_REWARD_SCALE:-1.0}"

TS="$(date +%Y%m%d_%H%M%S)"
ENV_TAG="${ENV_NAME//-/_}"
EXP_NAME="${EXP_NAME:-maze_hgdagger_${ENV_TAG}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs
mkdir -p "${MPLCONFIGDIR_VALUE}" "${WANDB_DIR_VALUE}" "${WANDB_CACHE_DIR_VALUE}"

ARGS=(
  --env_name "${ENV_NAME}"
  --num_envs "${NUM_ENVS}"
  --max_episode_steps "${MAX_EPISODE_STEPS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --seed "${SEED}"
  --device auto
  --train_render_mode none
  --obs_mode state
  --reward_type "${REWARD_TYPE}"
  --dense_reward_scale "${DENSE_REWARD_SCALE}"
  --use_intervention
  --intervention_mode "${INTERVENTION_MODE}"
  --teacher_type "${TEACHER_TYPE}"
  --tolerance_type "${TOLERANCE_TYPE}"
  --tolerance_value "${TOLERANCE_VALUE}"
  --intervention_episode_prob "${INTERVENTION_EPISODE_PROB}"
  --intervention_episode_prob_min "${INTERVENTION_EPISODE_PROB_MIN}"
  --intervention_episode_prob_decay_steps "${INTERVENTION_EPISODE_PROB_DECAY_STEPS}"
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --batch_size "${BATCH_SIZE}"
  --num_updates "${NUM_UPDATES}"
  --learning_starts "${LEARNING_STARTS}"
  --demo_buffer_capacity "${DEMO_BUFFER_CAPACITY}"
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
  --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --hg_ensemble_size "${HG_ENSEMBLE_SIZE}"
  --hg_doubt_percentile "${HG_DOUBT_PERCENTILE}"
  --project "${PROJECT}"
  --exp_name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY_VALUE}" ]]; then
  ARGS+=(--wandb_entity "${WANDB_ENTITY_VALUE}")
fi
if [[ -n "${WANDB_GROUP_VALUE}" ]]; then
  ARGS+=(--wandb_group "${WANDB_GROUP_VALUE}")
fi

if [[ "${HARD_BLOCK_LETHAL}" == "1" ]]; then
  ARGS+=(--hard_block_lethal)
fi
if [[ "${USE_LAYER_NORM}" == "1" ]]; then
  ARGS+=(--use_layer_norm)
fi
if [[ "${VIZ_ON_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--viz_on_checkpoint --viz_grid_resolution "${VIZ_GRID_RESOLUTION}" --viz_quiver_stride "${VIZ_QUIVER_STRIDE}")
fi
if [[ "${EVAL_SAVE_TRAJECTORY_PLOT}" == "1" ]]; then
  ARGS+=(--eval_save_trajectory_plot --eval_trajectory_plot_episodes "${EVAL_TRAJECTORY_PLOT_EPISODES}")
fi

echo "=== Starting ${EXP_NAME} (hgdagger) ==="
echo "Log file: ${LOG_FILE}"

MPLCONFIGDIR="${MPLCONFIGDIR_VALUE}" \
WANDB_DIR="${WANDB_DIR_VALUE}" \
WANDB_CACHE_DIR="${WANDB_CACHE_DIR_VALUE}" \
WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_hg_dagger_ogbench_maze.py \
  "${ARGS[@]}" \
  --use_wandb \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
