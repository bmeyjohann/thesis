#!/usr/bin/env bash
set -euo pipefail

# Two-run entropy ablation (task2) based on:
#   scripts/run_cube_triple_hinge_ls2k_dual_intervention.sh
#
# Keeps all core settings identical to the base launcher's active run:
# - learning_starts=2000
# - lagrangian preference loss, hinge violation
# - pref margin=0.01
# - linked preference sampling
# - component-threshold intervention
#
# Runs:
# 1) High/fixed entropy (alpha forced high)
# 2) Near-deterministic/no-entropy approximation (alpha forced ~0)
#
# Note: alpha cannot be exactly 0 via current CLI because temperature is
# represented in log-space and clamps require positive values. We use 1e-8.
#
# Optional overrides:
#   PROJECT=ogbench-manip-reward-debug NUM_ENVS=32 BATCH_SIZE=256 \
#   HIGH_ALPHA=0.10 LOW_ALPHA=1e-8 \
#   bash scripts/run_cube_triple_hinge_ls2k_entropy_ablation.sh

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
  echo "[run_cube_triple_hinge_ls2k_entropy_ablation] SAVE_INTERVAL=${SAVE_INTERVAL} is too low; clamping to 10000."
  SAVE_INTERVAL=10000
fi

PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.01}"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-1e-3}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"
PREF_VIOLATION_TARGET="${PREF_VIOLATION_TARGET:-0.0}"
PREF_LAGRANGIAN_VIOLATION_TYPE="${PREF_LAGRANGIAN_VIOLATION_TYPE:-hinge}"

TOL_XYZ="${TOL_XYZ:-0.35}"
TOL_YAW="${TOL_YAW:-0.45}"
TOL_GRIPPER="${TOL_GRIPPER:-0.90}"
TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

# Entropy settings.
HIGH_ALPHA="${HIGH_ALPHA:-0.10}"
LOW_ALPHA="${LOW_ALPHA:-1e-8}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_one() {
  local exp_name="$1"
  local alpha_value="$2"
  shift 2
  local log_file="${LOG_DIR}/${exp_name}.log"
  echo "=== Starting ${exp_name} (fixed alpha=${alpha_value}) ==="
  echo "Log file: ${log_file}"
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
    --alpha_min "${alpha_value}" \
    --alpha_max "${alpha_value}" \
    --alpha_freeze_steps 0 \
    --use_intervention \
    --intervention_mode agent \
    --teacher_type cube_markov \
    --intervention_episode_prob 1.0 \
    --intervention_episode_prob_min 1.0 \
    --intervention_episode_prob_decay_steps 0 \
    --tolerance_type component \
    --tolerance_xyz_value "${TOL_XYZ}" \
    --tolerance_yaw_value "${TOL_YAW}" \
    --tolerance_gripper_value "${TOL_GRIPPER}" \
    --tolerance_adaptive_enable \
    --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}" \
    --tolerance_adaptive_far_distance "${TOL_FAR_DIST}" \
    --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}" \
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
    --exp_name "${exp_name}" \
    "$@" \
    2>&1 | tee "${log_file}"
  echo "=== Finished ${exp_name} ==="
}

EXP_HIGH="${EXP_HIGH:-cube_triple_hinge_ls2k_entropy_high_${TS}}"
run_one "${EXP_HIGH}" "${HIGH_ALPHA}"

EXP_LOW="${EXP_LOW:-cube_triple_hinge_ls2k_entropy_zeroish_${TS}}"
run_one "${EXP_LOW}" "${LOW_ALPHA}"

echo "All runs completed."
