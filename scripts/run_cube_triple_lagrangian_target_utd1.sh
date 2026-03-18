#!/usr/bin/env bash
set -euo pipefail

# Single-run launcher for lagrangian target-dual experiment:
# - UTD = 1
# - learning_starts = 10000
# - larger batch / env count defaults
# - lambda init = 1.0
# - target-aware lagrangian dual update enabled
#
# Optional overrides:
#   PROJECT=ogbench-manip-reward-debug NUM_ENVS=64 BATCH_SIZE=1024 \
#   PREF_VIOLATION_TARGET=0.05 TOTAL_TIMESTEPS=300000 \
#   bash scripts/run_cube_triple_lagrangian_target_utd1.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-triple-singletask-task5-v0}"

NUM_CRITICS="${NUM_CRITICS:-5}"
NUM_ENVS="${NUM_ENVS:-64}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
LEARNING_STARTS="${LEARNING_STARTS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_UPDATES="${NUM_UPDATES:-1}"

INTERVENTION_PROB="${INTERVENTION_PROB:-1.0}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
if [[ "${SAVE_INTERVAL}" -lt 10000 ]]; then
  echo "[run_cube_triple_lagrangian_target_utd1] SAVE_INTERVAL=${SAVE_INTERVAL} is too low; clamping to 10000."
  SAVE_INTERVAL=10000
fi

PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.01}"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-1e-3}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"
PREF_VIOLATION_TARGET="${PREF_VIOLATION_TARGET:-0.05}"
PREF_LAGRANGIAN_VIOLATION_TYPE="${PREF_LAGRANGIAN_VIOLATION_TYPE:-smooth}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-cube_triple_linked_lagr_target_utd1_ls10k_ne64_bs1024_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
  --env_name "${ENV_NAME}" \
  --obs_mode state \
  --reward_type sparse \
  --cube_reward_mode sparse_intermediate \
  --include_relative_cube_features \
  --num_critics "${NUM_CRITICS}" \
  --num_envs "${NUM_ENVS}" \
  --train_render_mode none \
  --eval_render_mode none \
  --eval_num_envs "${EVAL_NUM_ENVS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${NUM_UPDATES}" \
  --use_intervention \
  --intervention_mode agent_always \
  --teacher_type cube_markov \
  --intervention_episode_prob "${INTERVENTION_PROB}" \
  --intervention_episode_prob_min "${INTERVENTION_PROB}" \
  --intervention_episode_prob_decay_steps 0 \
  --pref_sampling_mode linked \
  --pref_rank_weight "${PREF_RANK_WEIGHT}" \
  --pref_rank_margin "${PREF_RANK_MARGIN}" \
  --pref_stopgrad_positive \
  --pref_loss_type lagrangian \
  --pref_lambda_init "${PREF_LAMBDA_INIT}" \
  --pref_lambda_lr "${PREF_LAMBDA_LR}" \
  --pref_lambda_max "${PREF_LAMBDA_MAX}" \
  --pref_lambda_ema "${PREF_LAMBDA_EMA}" \
  --pref_violation_clip "${PREF_VIOLATION_CLIP}" \
  --pref_violation_target "${PREF_VIOLATION_TARGET}" \
  --pref_lagrangian_violation_type "${PREF_LAGRANGIAN_VIOLATION_TYPE}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}" \
  --use_wandb \
  --project "${PROJECT}" \
  --exp_name "${EXP_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
