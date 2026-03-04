#!/usr/bin/env bash
set -euo pipefail

# Single-cube deterministic test:
# - fixed reset seed each episode (static initial states)
# - demo buffer prefill
# - dense reward mode
#
# Optional overrides:
#   TOTAL_TIMESTEPS=80000 DEMO_PREFILL_STEPS=10000 bash scripts/run_cube_single_static_demo_dense_online.sh

ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"
PROJECT="${PROJECT:-ogbench_cube_debug}"
EXP_NAME="${EXP_NAME:-cube_single_static_demo_dense_$(date +%Y%m%d_%H%M%S)}"

NUM_ENVS="${NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

STATIC_RESET_SEED="${STATIC_RESET_SEED:-123}"
DEMO_PREFILL_STEPS="${DEMO_PREFILL_STEPS:-10000}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}.log"

echo "Starting ${EXP_NAME} -> ${LOG_FILE}"

WANDB_MODE=online \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
python train_fast_sac_ogbench_manip.py \
  --env_name "${ENV_NAME}" \
  --obs_mode state \
  --num_envs "${NUM_ENVS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}" \
  --reward_type sparse \
  --cube_reward_mode dense \
  --static_reset_seed "${STATIC_RESET_SEED}" \
  --demo_buffer_enable \
  --demo_buffer_capacity 100000 \
  --demo_prefill_steps "${DEMO_PREFILL_STEPS}" \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent \
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}" \
  --teacher_type cube_plan \
  --tolerance_type l2 \
  --tolerance_value 0.05 \
  --use_wandb \
  --project "${PROJECT}" \
  --exp_name "${EXP_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "Done: ${EXP_NAME}"
