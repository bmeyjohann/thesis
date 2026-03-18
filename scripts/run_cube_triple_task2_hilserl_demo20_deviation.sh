#!/usr/bin/env bash
set -euo pipefail

# HILSERL-style baseline for cube-triple task2:
# - demo buffer enabled and prefilled with 20 teacher demo episodes
# - deviation-based intervention/replacement during online training
# - NO preference learning (no pref buffer, no pref loss contribution)
#
# Optional overrides:
#   PROJECT=ogbench-manip-reward-debug TOTAL_TIMESTEPS=300000 \
#   bash scripts/run_cube_triple_task2_hilserl_demo20_deviation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-triple-singletask-task2-v0}"

NUM_CRITICS="${NUM_CRITICS:-5}"
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
if [[ "${SAVE_INTERVAL}" -lt 10000 ]]; then
  echo "[run_cube_triple_task2_hilserl_demo20_deviation] SAVE_INTERVAL=${SAVE_INTERVAL} is too low; clamping to 10000."
  SAVE_INTERVAL=10000
fi

# Deviation-based intervention thresholds.
TOL_XYZ="${TOL_XYZ:-0.35}"
TOL_YAW="${TOL_YAW:-0.45}"
TOL_GRIPPER="${TOL_GRIPPER:-0.90}"
TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

# Demo buffer setup.
DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-cube_triple_task2_hilserl_demo20_dev_no_pref_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
  --env_name "${ENV_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --device auto \
  --train_render_mode none \
  --obs_mode state \
  --include_relative_cube_features \
  --reward_type sparse \
  --cube_reward_mode sparse_intermediate \
  --num_critics "${NUM_CRITICS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${NUM_UPDATES}" \
  --learning_starts "${LEARNING_STARTS}" \
  --alpha_min 0.0 \
  --alpha_max 1.0 \
  --alpha_freeze_steps 0 \
  --use_intervention \
  --intervention_mode agent \
  --teacher_type cube_markov \
  --intervention_episode_prob 1.0 \
  --intervention_episode_prob_min 1.0 \
  --intervention_episode_prob_decay_steps 0 \
  --tolerance_type component \
  --tolerance_value 30.0 \
  --tolerance_xyz_value "${TOL_XYZ}" \
  --tolerance_yaw_value "${TOL_YAW}" \
  --tolerance_gripper_value "${TOL_GRIPPER}" \
  --tolerance_adaptive_enable \
  --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}" \
  --tolerance_adaptive_far_distance "${TOL_FAR_DIST}" \
  --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}" \
  --hard_block_lethal \
  --demo_buffer_enable \
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}" \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent_always \
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}" \
  --pref_sample_ratio 0.0 \
  --pref_rank_weight 0.0 \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --eval_num_envs "${EVAL_NUM_ENVS}" \
  --eval_render_mode none \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval 64 \
  --use_wandb \
  --project "${PROJECT}" \
  --exp_name "${EXP_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="

