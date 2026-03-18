#!/usr/bin/env bash
set -euo pipefail

# Two-run launcher for cube triple task2 that mirrors
# cube_triple_task2_hinge_m001_ls2k_teacher_component_* settings
# but uses dense cube reward mode.
# Experiment 1: deviation-threshold based (`intervention_mode=agent`,
# `tolerance_type=component`) with full episode gating by default.
# Experiment 2: teacher always active (`intervention_mode=agent_always`).
#
# Optional overrides:
#   PROJECT=ogbench-manip-reward-debug TOTAL_TIMESTEPS=300000 \
#   bash scripts/run_cube_triple_task2_dense_component_deviation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-triple-singletask-task2-v0}"

NUM_CRITICS="${NUM_CRITICS:-5}"
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
if [[ "${SAVE_INTERVAL}" -lt 10000 ]]; then
  echo "[run_cube_triple_task2_dense_component_deviation] SAVE_INTERVAL=${SAVE_INTERVAL} is too low; clamping to 10000."
  SAVE_INTERVAL=10000
fi

INTERVENTION_PROB="${INTERVENTION_PROB:-1.0}"

TOL_XYZ="${TOL_XYZ:-0.35}"
TOL_YAW="${TOL_YAW:-0.45}"
TOL_GRIPPER="${TOL_GRIPPER:-0.90}"
TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_DEV="${EXP_DEV:-cube_triple_task2_hinge_m001_ls2k_teacher_component_dense_dev_${TS}}"
EXP_ALWAYS="${EXP_ALWAYS:-cube_triple_task2_hinge_m001_ls2k_teacher_always_dense_${TS}}"
mkdir -p logs

run_one() {
  local exp_name="$1"
  shift
  local log_file="logs/${exp_name}.log"
  echo "=== Starting ${exp_name} ==="
  echo "Log file: ${log_file}"
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
    --cube_reward_mode dense \
    --use_intervention \
    --teacher_type cube_markov \
    --hard_block_lethal \
    --num_critics "${NUM_CRITICS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_updates "${NUM_UPDATES}" \
    --learning_starts "${LEARNING_STARTS}" \
    --alpha_min 0.0 \
    --alpha_max 1.0 \
    --alpha_freeze_steps 0 \
    --pref_sampling_mode linked \
    --pref_rank_weight 1.0 \
    --pref_rank_margin 0.01 \
    --pref_stopgrad_positive \
    --pref_loss_type lagrangian \
    --pref_lambda_init 1.0 \
    --pref_lambda_lr 1e-3 \
    --pref_lambda_max 10.0 \
    --pref_lambda_ema 0.9 \
    --pref_violation_clip 10.0 \
    --pref_violation_target 0.0 \
    --pref_lagrangian_violation_type hinge \
    --intervention_episode_prob "${INTERVENTION_PROB}" \
    --intervention_episode_prob_min "${INTERVENTION_PROB}" \
    --intervention_episode_prob_decay_steps 0 \
    --eval_interval "${EVAL_INTERVAL}" \
    --num_eval_episodes "${NUM_EVAL_EPISODES}" \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    --eval_render_mode none \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval 64 \
    --use_wandb \
    --project "${PROJECT}" \
    --exp_name "${exp_name}" \
    "$@" \
    2>&1 | tee "${log_file}"
  echo "=== Finished ${exp_name} ==="
}

run_one "${EXP_DEV}" \
  --intervention_mode agent \
  --tolerance_type component \
  --tolerance_value 30.0 \
  --tolerance_xyz_value "${TOL_XYZ}" \
  --tolerance_yaw_value "${TOL_YAW}" \
  --tolerance_gripper_value "${TOL_GRIPPER}" \
  --tolerance_adaptive_enable \
  --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}" \
  --tolerance_adaptive_far_distance "${TOL_FAR_DIST}" \
  --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}"

run_one "${EXP_ALWAYS}" \
  --intervention_mode agent_always

echo "All runs completed."
