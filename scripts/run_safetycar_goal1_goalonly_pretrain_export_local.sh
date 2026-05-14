#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

TS="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-safetycar_goal1_goalonly_small_ln_pretrain_${TS}}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/datasets/safetygym/${EXP_NAME}_replay.npz}"

echo "Starting SafetyCar Goal1 goal-only pretrain/export"
echo "run:     ${EXP_NAME}"
echo "dataset: ${DATASET_PATH}"

WANDB_MODE="${WANDB_MODE:-online}" \
PROJECT="${PROJECT:-thesis-safetygym}" \
WANDB_GROUP="${WANDB_GROUP:-safetycar_goalonly_pretrain_20260511}" \
EXP_NAME="$EXP_NAME" \
ENV_NAME="${ENV_NAME:-SafetyCarGoal1-v0}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}" \
LEARNING_STARTS="${LEARNING_STARTS:-5000}" \
VARIANT=plain \
MODULE_IMPL=custom \
USE_LAYER_NORM=1 \
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}" \
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}" \
REWARD_MODE="${REWARD_MODE:-dense}" \
STEP_PENALTY="${STEP_PENALTY:--0.001}" \
COST_PENALTY=0.0 \
OBS_MASK_MODE=goal_only_lidar \
TERMINATE_ON_GOAL=1 \
RESEED_ON_EPISODE_RESET=0 \
CAR_ACTION_MODE=raw_wheels \
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}" \
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}" \
OBS_NORMALIZATION=1 \
SCALE_ACTOR_TO_ENV_BOUNDS=1 \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}" \
LOG_INTERVAL="${LOG_INTERVAL:-1000}" \
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}" \
VIZ_ON_CHECKPOINT="${VIZ_ON_CHECKPOINT:-1}" \
EVAL_SAVE_EPISODE_PLOTS="${EVAL_SAVE_EPISODE_PLOTS:-1}" \
EXPORT_FINAL_REPLAY_DATASET=1 \
EXPORT_FINAL_REPLAY_DATASET_PATH="$DATASET_PATH" \
EXPORT_REPLAY_DATASET_LABEL="${EXPORT_REPLAY_DATASET_LABEL:-goalonly_pretrain_replay}" \
/home/benjamin/thesis/scripts/run_safetycar_minimal_human_ready_local.sh
