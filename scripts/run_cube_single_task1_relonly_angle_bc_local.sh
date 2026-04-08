#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"
REWARD_TYPE="${REWARD_TYPE:-sparse}"
CUBE_REWARD_MODE="${CUBE_REWARD_MODE:-dense}"
INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-1.0}"
GAMMA="${GAMMA:-0.97}"

NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
NUM_UPDATES="${NUM_UPDATES:-7}"
BC_COLLECT_ONLINE_TEACHER_DATA="${BC_COLLECT_ONLINE_TEACHER_DATA:-0}"

DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-20}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"

TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

DISABLE_ROTATION="${DISABLE_ROTATION:-1}"
NAME_SUFFIX="${NAME_SUFFIX:-}"

TS="$(date +%Y%m%d_%H%M%S)"
ROT_TAG="rot"
if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  ROT_TAG="norot"
fi

BC_TAG="bc_demoonly"
if [[ "${BC_COLLECT_ONLINE_TEACHER_DATA}" == "1" ]]; then
  BC_TAG="bc_onlineagg"
fi
BASE_NAME="cube_single_task1_relonly_anglebaseline_${ROT_TAG}_${BC_TAG}_interv_${INTERVENTION_EPISODE_PROB}"
if [[ -n "${NAME_SUFFIX}" ]]; then
  BASE_NAME="${BASE_NAME}_${NAME_SUFFIX}"
fi
EXP_NAME="${EXP_NAME:-${BASE_NAME}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "WANDB mode: ${WANDB_MODE_VALUE}"
echo "Teacher demo prefill: ${DEMO_PREFILL_EPISODES} episodes into expert dataset"
echo "Intervention episode probability: ${INTERVENTION_EPISODE_PROB}"
echo "Gamma: ${GAMMA}"
echo "UTD (num_updates): ${NUM_UPDATES}"
echo "Disable rotation: ${DISABLE_ROTATION}"
echo "Online teacher aggregation: ${BC_COLLECT_ONLINE_TEACHER_DATA}"

ARGS=(
  --env_name "${ENV_NAME}"
  --num_envs "${NUM_ENVS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --device auto
  --train_render_mode none
  --obs_mode state
  --no_include_goal
  --include_relative_cube_features
  --relative_only_obs
  --reward_type "${REWARD_TYPE}"
  --cube_reward_mode "${CUBE_REWARD_MODE}"
  --use_intervention
  --teacher_type cube_markov
  --hard_block_lethal
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --batch_size "${BATCH_SIZE}"
  --num_updates "${NUM_UPDATES}"
  --learning_starts "${LEARNING_STARTS}"
  --gamma "${GAMMA}"
  --use_layer_norm
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
  --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
  --intervention_episode_prob "${INTERVENTION_EPISODE_PROB}"
  --intervention_episode_prob_min "${INTERVENTION_EPISODE_PROB}"
  --intervention_episode_prob_decay_steps 0
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --eval_render_mode none
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --use_wandb
  --project "${PROJECT}"
  --exp_name "${EXP_NAME}"
  --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}"
  --tolerance_adaptive_far_distance "${TOL_FAR_DIST}"
  --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}"
)

if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  ARGS+=(--disable_rotation)
fi
if [[ "${BC_COLLECT_ONLINE_TEACHER_DATA}" == "1" ]]; then
  ARGS+=(--bc_collect_online_teacher_data)
fi

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_behavior_cloning_ogbench_manip.py \
  "${ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
