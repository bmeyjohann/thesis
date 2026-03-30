#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"
VR_HOST="${VR_HOST:-192.168.2.182}"
VR_PORT="${VR_PORT:-8765}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_CRITICS="${NUM_CRITICS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"

CTA_RATIO="${CTA_RATIO:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-cube_single_relonly_vr_own_demo20_cta${CTA_RATIO}__num_updates${NUM_UPDATES}__${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "VR endpoint: ${VR_HOST}:${VR_PORT}"
echo "CTA ratio: ${CTA_RATIO}"
echo "Number of updates: ${NUM_UPDATES}"

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
  --env_name "${ENV_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --device auto \
  --train_render_mode human \
  --obs_mode state \
  --no_include_goal \
  --include_relative_cube_features \
  --relative_only_obs \
  --reward_type sparse \
  --cube_reward_mode dense \
  --use_intervention \
  --intervention_mode human \
  --human_input_device vr \
  --vr_mode connect \
  --vr_host "${VR_HOST}" \
  --vr_port "${VR_PORT}" \
  --num_critics "${NUM_CRITICS}" \
  --actor_hidden_dim 256 \
  --critic_hidden_dim 512 \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${NUM_UPDATES}" \
  --learning_starts "${LEARNING_STARTS}" \
  --alpha_min 0.0 \
  --alpha_max 1.0 \
  --alpha_freeze_steps 0 \
  --use_layer_norm \
  --demo_buffer_enable \
  --demo_sample_ratio 0.5 \
  --demo_dataset_auto_load \
  --demo_dataset_target demo \
  --pref_buffer_enable \
  --pref_sampling_mode linked \
  --pref_sample_ratio 0.5 \
  --pref_rank_weight 1.0 \
  --pref_rank_margin 0.01 \
  --pref_loss_type lagrangian \
  --pref_lambda_init 1.0 \
  --pref_lambda_lr 1e-3 \
  --pref_lambda_max 10.0 \
  --pref_lambda_ema 0.9 \
  --pref_violation_clip 10.0 \
  --pref_violation_target 0.0 \
  --pref_lagrangian_violation_type hinge \
  --pref_stopgrad_positive \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --eval_num_envs "${EVAL_NUM_ENVS}" \
  --eval_render_mode none \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --use_wandb \
  --project "${PROJECT}" \
  --exp_name "${EXP_NAME}" \
  --tolerance_adaptive_near_distance 0.08 \
  --tolerance_adaptive_far_distance 0.30 \
  --tolerance_adaptive_near_scale 0.35 \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
