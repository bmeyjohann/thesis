#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-triple-singletask-task2-v0}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-20}"
INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-0.5}"

NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
NUM_UPDATES="${NUM_UPDATES:-1}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"

TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-cube_triple_task2_relonly_layernorm_newteacher_utd1_60k_hilserl_interv_${INTERVENTION_EPISODE_PROB}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "WANDB mode: ${WANDB_MODE_VALUE}"
echo "Teacher demo prefill: ${DEMO_PREFILL_EPISODES} episodes into demo buffer"
echo "Sampling replay/demo ratio: 50/50 (demo_sample_ratio=${DEMO_SAMPLE_RATIO})"
echo "Intervention episode probability: ${INTERVENTION_EPISODE_PROB}"

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
  --no_include_goal \
  --include_relative_cube_features \
  --relative_only_obs \
  --reward_type sparse \
  --cube_reward_mode dense \
  --use_intervention \
  --teacher_type cube_markov \
  --hard_block_lethal \
  --num_critics "${NUM_CRITICS}" \
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}" \
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}" \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${NUM_UPDATES}" \
  --learning_starts "${LEARNING_STARTS}" \
  --alpha_min 0.0 \
  --alpha_max 1.0 \
  --alpha_freeze_steps 0 \
  --use_layer_norm \
  --demo_buffer_enable \
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}" \
  --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}" \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent_always \
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}" \
  --store_intervened_in_demo_buffer \
  --pref_sample_ratio 0.0 \
  --pref_rank_weight 0.0 \
  --intervention_episode_prob "${INTERVENTION_EPISODE_PROB}" \
  --intervention_episode_prob_min "${INTERVENTION_EPISODE_PROB}" \
  --intervention_episode_prob_decay_steps 0 \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --eval_num_envs "${EVAL_NUM_ENVS}" \
  --eval_render_mode none \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --use_wandb \
  --project "${PROJECT}" \
  --exp_name "${EXP_NAME}" \
  --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}" \
  --tolerance_adaptive_far_distance "${TOL_FAR_DIST}" \
  --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
