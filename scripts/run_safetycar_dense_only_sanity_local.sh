#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-1}"

RENDER_MODE="${RENDER_MODE:-none}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-1.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-1.0}"
STEP_PENALTY="${STEP_PENALTY:-0.0}"
REWARD_MODE="${REWARD_MODE:-dense}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-safetycar_goal2_dense_only_sanity_${TS}}"
EXP_NAME="${EXP_NAME:-${WANDB_RUN_NAME}}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"

echo "Starting SafetyCar dense-only sanity run."
echo "Current SafetyGym trainer is single-env, so this launcher pins --num_envs 1."
echo "Human intervention is disabled."
echo "Preference loss is disabled."

conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" train_fast_sac_safetygym.py \
  --env_name "${ENV_NAME}" \
  --exp_name "${EXP_NAME}" \
  --render_mode "${RENDER_MODE}" \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --num_envs 1 \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --update_every "${UPDATE_EVERY}" \
  --updates_per_cycle "${UPDATES_PER_CYCLE}" \
  --reward_mode "${REWARD_MODE}" \
  --step_penalty "${STEP_PENALTY}" \
  --pref_rank_weight 0.0 \
  --demo_sample_ratio 0.0 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE_VALUE}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}"
