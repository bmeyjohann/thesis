#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-1}"

RENDER_MODE="${RENDER_MODE:-none}"
ENV_FPS_LIMIT="${ENV_FPS_LIMIT:-0}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-1.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-1.0}"

NUM_CRITICS="${NUM_CRITICS:-2}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
TS="$(date +%Y%m%d_%H%M%S)"

DENSE_PLUS_SPARSE_RUN_NAME="${DENSE_PLUS_SPARSE_RUN_NAME:-safetycar_goal2_dense_plus_sparse_diag_${TS}}"
DENSE_PLUS_SPARSE_EXP_NAME="${DENSE_PLUS_SPARSE_EXP_NAME:-${DENSE_PLUS_SPARSE_RUN_NAME}}"

DENSE_ONLY_RUN_NAME="${DENSE_ONLY_RUN_NAME:-safetycar_goal2_dense_only_diag_${TS}}"
DENSE_ONLY_EXP_NAME="${DENSE_ONLY_EXP_NAME:-${DENSE_ONLY_RUN_NAME}}"

echo "Starting SafetyCar reward comparison runs."
echo "Run 1/2: dense_plus_sparse"
echo "Run 2/2: dense"
echo "Current SafetyGym trainer is single-env, so this launcher pins --num_envs 1."

conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" train_fast_sac_safetygym.py \
  --env_name "${ENV_NAME}" \
  --exp_name "${DENSE_PLUS_SPARSE_EXP_NAME}" \
  --render_mode "${RENDER_MODE}" \
  --env_fps_limit "${ENV_FPS_LIMIT}" \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --num_envs 1 \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --update_every "${UPDATE_EVERY}" \
  --updates_per_cycle "${UPDATES_PER_CYCLE}" \
  --reward_mode dense_plus_sparse \
  --pref_rank_weight 0.0 \
  --demo_sample_ratio 0.0 \
  --num_critics "${NUM_CRITICS}" \
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}" \
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}" \
  --use_layer_norm \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE_VALUE}" \
  --wandb_run_name "${DENSE_PLUS_SPARSE_RUN_NAME}" \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}"

conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" train_fast_sac_safetygym.py \
  --env_name "${ENV_NAME}" \
  --exp_name "${DENSE_ONLY_EXP_NAME}" \
  --render_mode "${RENDER_MODE}" \
  --env_fps_limit "${ENV_FPS_LIMIT}" \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --num_envs 1 \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --update_every "${UPDATE_EVERY}" \
  --updates_per_cycle "${UPDATES_PER_CYCLE}" \
  --reward_mode dense \
  --pref_rank_weight 0.0 \
  --demo_sample_ratio 0.0 \
  --num_critics "${NUM_CRITICS}" \
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}" \
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}" \
  --use_layer_norm \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE_VALUE}" \
  --wandb_run_name "${DENSE_ONLY_RUN_NAME}" \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}"
