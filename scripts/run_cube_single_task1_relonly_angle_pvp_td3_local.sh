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

NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
LEARNING_STARTS="${LEARNING_STARTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
CRITIC_LR="${CRITIC_LR:-1e-4}"

PVP_POLICY_DELAY="${PVP_POLICY_DELAY:-2}"
PVP_TARGET_POLICY_NOISE="${PVP_TARGET_POLICY_NOISE:-0.2}"
PVP_TARGET_NOISE_CLIP="${PVP_TARGET_NOISE_CLIP:-0.5}"
PVP_CQL_COEFFICIENT="${PVP_CQL_COEFFICIENT:-1.0}"
PVP_PROXY_VALUE_BOUND="${PVP_PROXY_VALUE_BOUND:-1.0}"
PVP_INCLUDE_ENV_REWARD_IN_TD="${PVP_INCLUDE_ENV_REWARD_IN_TD:-0}"
PVP_BALANCE_SAMPLE="${PVP_BALANCE_SAMPLE:-1}"
PVP_STOP_TD_ON_INTERVENTION_START="${PVP_STOP_TD_ON_INTERVENTION_START:-1}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"

TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

DISABLE_ROTATION="${DISABLE_ROTATION:-1}"
NAME_SUFFIX="${NAME_SUFFIX:-faithful}"

TS="$(date +%Y%m%d_%H%M%S)"
ROT_TAG="rot"
if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  ROT_TAG="norot"
fi

BASE_NAME="cube_single_task1_relonly_anglebaseline_${ROT_TAG}_pvp_td3_interv_${INTERVENTION_EPISODE_PROB}"
if [[ -n "${NAME_SUFFIX}" ]]; then
  BASE_NAME="${BASE_NAME}_${NAME_SUFFIX}"
fi
EXP_NAME="${EXP_NAME:-${BASE_NAME}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "WANDB mode: ${WANDB_MODE_VALUE}"
echo "Faithful PVP-TD3 with balanced novice/human buffers"
echo "Intervention episode probability: ${INTERVENTION_EPISODE_PROB}"
echo "Rotation disabled: ${DISABLE_ROTATION}"

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
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}"
  --batch_size "${BATCH_SIZE}"
  --num_updates "${NUM_UPDATES}"
  --learning_starts "${LEARNING_STARTS}"
  --gamma "${GAMMA}"
  --tau "${TAU}"
  --actor_learning_rate "${ACTOR_LR}"
  --critic_learning_rate "${CRITIC_LR}"
  --demo_buffer_enable
  --demo_prefill_episodes 0
  --demo_prefill_num_envs 0
  --demo_sample_ratio 0.5
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
  --pvp_policy_delay "${PVP_POLICY_DELAY}"
  --pvp_target_policy_noise "${PVP_TARGET_POLICY_NOISE}"
  --pvp_target_noise_clip "${PVP_TARGET_NOISE_CLIP}"
  --pvp_cql_coefficient "${PVP_CQL_COEFFICIENT}"
  --pvp_proxy_value_bound "${PVP_PROXY_VALUE_BOUND}"
  --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}"
  --tolerance_adaptive_far_distance "${TOL_FAR_DIST}"
  --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}"
)

if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  ARGS+=(--disable_rotation)
fi
if [[ "${PVP_INCLUDE_ENV_REWARD_IN_TD}" == "1" ]]; then
  ARGS+=(--pvp_include_env_reward_in_td)
fi
if [[ "${PVP_BALANCE_SAMPLE}" == "0" ]]; then
  ARGS+=(--no_pvp_balance_sample)
fi
if [[ "${PVP_STOP_TD_ON_INTERVENTION_START}" == "0" ]]; then
  ARGS+=(--no_pvp_stop_td_on_intervention_start)
fi

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_pvp_td3_ogbench_manip.py \
  "${ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
