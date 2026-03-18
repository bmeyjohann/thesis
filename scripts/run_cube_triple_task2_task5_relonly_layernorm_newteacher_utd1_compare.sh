#!/usr/bin/env bash
set -euo pipefail

# Two-run comparison:
# - task2 and task5
# - relative-only manipulation observations (compact proprio + relative features)
# - no absolute goal concatenation
# - LayerNorm enabled
# - UTD fixed to 1 for both runs
# - shorter horizon for teacher-comparison triage

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"

NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
NUM_UPDATES="${NUM_UPDATES:-1}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
if [[ "${SAVE_INTERVAL}" -lt 10000 ]]; then
  echo "[run_cube_triple_task2_task5_relonly_layernorm_newteacher_utd1_compare] SAVE_INTERVAL=${SAVE_INTERVAL} is too low; clamping to 10000."
  SAVE_INTERVAL=10000
fi

TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_TASK2="${EXP_TASK2:-cube_triple_task2_relonly_layernorm_newteacher_utd1_60k_${TS}}"
EXP_TASK5="${EXP_TASK5:-cube_triple_task5_relonly_layernorm_newteacher_utd1_60k_${TS}}"

mkdir -p logs

run_task2() {
  local log_file="logs/${EXP_TASK2}.log"
  echo "=== Starting ${EXP_TASK2} ==="
  echo "Log file: ${log_file}"

  WANDB_MODE="${WANDB_MODE_VALUE}" \
  WANDB_CONSOLE=off \
  WANDB_SILENT=true \
  "${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
    --env_name cube-triple-singletask-task2-v0 \
    --num_envs "${NUM_ENVS}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --device auto \
    --train_render_mode none \
    --obs_mode state \
    --no_include_goal \
    --include_relative_cube_features \
    --relative_only_obs \
    --reward_type sparse \
    --cube_reward_mode dense \
    --use_intervention \
    --teacher_type cube_markov \
    --hard_block_lethal \
    --num_critics 2 \
    --actor_hidden_dim 256 \
    --critic_hidden_dim 512 \
    --batch_size "${BATCH_SIZE}" \
    --num_updates "${NUM_UPDATES}" \
    --learning_starts "${LEARNING_STARTS}" \
    --alpha_min 0.0 \
    --alpha_max 1.0 \
    --alpha_freeze_steps 0 \
    --use_layer_norm \
    --pref_sampling_mode linked \
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
    --intervention_episode_prob 1.0 \
    --intervention_episode_prob_min 1.0 \
    --intervention_episode_prob_decay_steps 0 \
    --eval_interval "${EVAL_INTERVAL}" \
    --num_eval_episodes "${NUM_EVAL_EPISODES}" \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    --eval_render_mode none \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval 64 \
    --use_wandb \
    --project "${PROJECT}" \
    --exp_name "${EXP_TASK2}" \
    --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}" \
    --tolerance_adaptive_far_distance "${TOL_FAR_DIST}" \
    --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}" \
    2>&1 | tee "${log_file}"

  echo "=== Finished ${EXP_TASK2} ==="
}

run_task5() {
  local log_file="logs/${EXP_TASK5}.log"
  echo "=== Starting ${EXP_TASK5} ==="
  echo "Log file: ${log_file}"

  WANDB_MODE="${WANDB_MODE_VALUE}" \
  WANDB_CONSOLE=off \
  WANDB_SILENT=true \
  "${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
    --env_name cube-triple-singletask-task5-v0 \
    --num_envs "${NUM_ENVS}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --device auto \
    --train_render_mode none \
    --obs_mode state \
    --no_include_goal \
    --include_relative_cube_features \
    --relative_only_obs \
    --reward_type sparse \
    --cube_reward_mode sparse_intermediate \
    --use_intervention \
    --teacher_type cube_markov \
    --hard_block_lethal \
    --num_critics 5 \
    --batch_size "${BATCH_SIZE}" \
    --num_updates "${NUM_UPDATES}" \
    --learning_starts "${LEARNING_STARTS}" \
    --alpha_min 0.0 \
    --alpha_max 1.0 \
    --alpha_freeze_steps 0 \
    --use_layer_norm \
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
    --intervention_episode_prob 1.0 \
    --intervention_episode_prob_min 1.0 \
    --intervention_episode_prob_decay_steps 0 \
    --eval_interval "${EVAL_INTERVAL}" \
    --num_eval_episodes "${NUM_EVAL_EPISODES}" \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    --eval_render_mode none \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval 64 \
    --use_wandb \
    --project "${PROJECT}" \
    --exp_name "${EXP_TASK5}" \
    --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}" \
    --tolerance_adaptive_far_distance "${TOL_FAR_DIST}" \
    --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}" \
    2>&1 | tee "${log_file}"

  echo "=== Finished ${EXP_TASK5} ==="
}

run_task2
run_task5
