#!/usr/bin/env bash
set -euo pipefail

# Allow the experiment queue to pass KEY=VALUE overrides as positional args.
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
ALGO_VARIANT="${ALGO_VARIANT:-own}"
REWARD_TYPE="${REWARD_TYPE:-sparse}"
CUBE_REWARD_MODE="${CUBE_REWARD_MODE:-dense}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-20}"
INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-1.0}"
GAMMA="${GAMMA:-0.97}"

NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
NUM_UPDATES="${NUM_UPDATES:-1}"
CTA_RATIO="${CTA_RATIO:-1}"
ALPHA_INIT="${ALPHA_INIT:-0.001}"
FIXED_ALPHA="${FIXED_ALPHA:--1}"
ALPHA_UPDATE_STUDENT_ONLY="${ALPHA_UPDATE_STUDENT_ONLY:-0}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"
ALPHA_FREEZE_STEPS="${ALPHA_FREEZE_STEPS:-0}"

PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.01}"
PREF_CRITIC_SCOPE="${PREF_CRITIC_SCOPE:-all}"
PREF_LOSS_TYPE="${PREF_LOSS_TYPE:-lagrangian}"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-1e-3}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"
PREF_VIOLATION_TARGET="${PREF_VIOLATION_TARGET:-0.0}"
PREF_LAGRANGIAN_VIOLATION_TYPE="${PREF_LAGRANGIAN_VIOLATION_TYPE:-hinge}"
PREF_STOPGRAD_POSITIVE="${PREF_STOPGRAD_POSITIVE:-1}"
STORE_INTERVENED_IN_DEMO_BUFFER="${STORE_INTERVENED_IN_DEMO_BUFFER:-0}"
PVP_PROXY_VALUE_BOUND="${PVP_PROXY_VALUE_BOUND:-1.0}"
PVP_INCLUDE_ENV_REWARD_IN_TD="${PVP_INCLUDE_ENV_REWARD_IN_TD:-0}"
EIL_THRESHOLD="${EIL_THRESHOLD:-0.0}"
EIL_GOOD_MARGIN="${EIL_GOOD_MARGIN:-0.0}"
EIL_BAD_MARGIN="${EIL_BAD_MARGIN:-0.01}"
EIL_PAIR_MARGIN="${EIL_PAIR_MARGIN:-0.01}"
EIL_BAD_PRE_STEPS="${EIL_BAD_PRE_STEPS:-8}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"
TRAIN_RENDER_MODE="${TRAIN_RENDER_MODE:-none}"
PROFILE_TIMING="${PROFILE_TIMING:-0}"

TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"

DISABLE_ROTATION="${DISABLE_ROTATION:-0}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
EXPORT_REPLAY_DATASET_INTERVAL="${EXPORT_REPLAY_DATASET_INTERVAL:-0}"
EXPORT_REPLAY_DATASET_PATH="${EXPORT_REPLAY_DATASET_PATH:-}"
EXPORT_REPLAY_DATASET_DIR="${EXPORT_REPLAY_DATASET_DIR:-}"
EXPORT_REPLAY_DATASET_LABEL="${EXPORT_REPLAY_DATASET_LABEL:-}"
EXPORT_REPLAY_DATASET_MAX_ROWS="${EXPORT_REPLAY_DATASET_MAX_ROWS:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
ROT_TAG="rot"
if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  ROT_TAG="norot"
fi

BASE_NAME="cube_single_task1_relonly_anglebaseline_${ROT_TAG}_utd${NUM_UPDATES}_cta${CTA_RATIO}_interv_${INTERVENTION_EPISODE_PROB}"
if [[ -n "${NAME_SUFFIX}" ]]; then
  BASE_NAME="${BASE_NAME}_${NAME_SUFFIX}"
fi
EXP_NAME="${EXP_NAME:-${BASE_NAME}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "WANDB mode: ${WANDB_MODE_VALUE}"
echo "Teacher demo prefill: ${DEMO_PREFILL_EPISODES} episodes into demo buffer"
echo "Sampling replay/demo ratio: 50/50 (demo_sample_ratio=${DEMO_SAMPLE_RATIO})"
echo "Algo variant: ${ALGO_VARIANT}"
echo "Reward type / cube reward mode: ${REWARD_TYPE} / ${CUBE_REWARD_MODE}"
echo "Intervention episode probability: ${INTERVENTION_EPISODE_PROB}"
echo "Gamma: ${GAMMA}"
echo "CTA ratio: ${CTA_RATIO}"
echo "UTD (num_updates): ${NUM_UPDATES}"
echo "Disable rotation: ${DISABLE_ROTATION}"
echo "Alpha init: ${ALPHA_INIT}"
echo "Fixed alpha: ${FIXED_ALPHA}"
echo "Alpha update student-only: ${ALPHA_UPDATE_STUDENT_ONLY}"
echo "Alpha min/max/freeze: ${ALPHA_MIN}/${ALPHA_MAX}/${ALPHA_FREEZE_STEPS}"
echo "Pref critic scope: ${PREF_CRITIC_SCOPE}"
echo "Pref loss type: ${PREF_LOSS_TYPE}"
echo "Pref stopgrad positive: ${PREF_STOPGRAD_POSITIVE}"
echo "Store intervened in demo buffer: ${STORE_INTERVENED_IN_DEMO_BUFFER}"
echo "PVP include env reward in TD: ${PVP_INCLUDE_ENV_REWARD_IN_TD}"
echo "EIL threshold/good/bad/pair/pre: ${EIL_THRESHOLD}/${EIL_GOOD_MARGIN}/${EIL_BAD_MARGIN}/${EIL_PAIR_MARGIN}/${EIL_BAD_PRE_STEPS}"
echo "Replay export interval: ${EXPORT_REPLAY_DATASET_INTERVAL}"
echo "Train render mode: ${TRAIN_RENDER_MODE}"
echo "Profile timing: ${PROFILE_TIMING}"

ARGS=(
  --env_name "${ENV_NAME}"
  --algo_variant "${ALGO_VARIANT}"
  --num_envs "${NUM_ENVS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --device auto
  --train_render_mode "${TRAIN_RENDER_MODE}"
  --obs_mode state
  --no_include_goal
  --include_relative_cube_features
  --relative_only_obs
  --reward_type "${REWARD_TYPE}"
  --cube_reward_mode "${CUBE_REWARD_MODE}"
  --pvp_proxy_value_bound "${PVP_PROXY_VALUE_BOUND}"
  --eil_threshold "${EIL_THRESHOLD}"
  --eil_good_margin "${EIL_GOOD_MARGIN}"
  --eil_bad_margin "${EIL_BAD_MARGIN}"
  --eil_pair_margin "${EIL_PAIR_MARGIN}"
  --eil_bad_pre_steps "${EIL_BAD_PRE_STEPS}"
  --use_intervention
  --teacher_type cube_markov
  --hard_block_lethal
  --num_critics "${NUM_CRITICS}"
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}"
  --batch_size "${BATCH_SIZE}"
  --num_updates "${NUM_UPDATES}"
  --cta_ratio "${CTA_RATIO}"
  --learning_starts "${LEARNING_STARTS}"
  --gamma "${GAMMA}"
  --alpha_init "${ALPHA_INIT}"
  --fixed_alpha "${FIXED_ALPHA}"
  --alpha_min "${ALPHA_MIN}"
  --alpha_max "${ALPHA_MAX}"
  --alpha_freeze_steps "${ALPHA_FREEZE_STEPS}"
  --use_layer_norm
  --demo_buffer_enable
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
  --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
  --demo_prefill_target demo
  --demo_prefill_intervention_mode agent_always
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
  --pref_buffer_enable
  --pref_sampling_mode linked
  --pref_sample_ratio 0.0
  --pref_rank_weight "${PREF_RANK_WEIGHT}"
  --pref_rank_margin "${PREF_RANK_MARGIN}"
  --pref_critic_scope "${PREF_CRITIC_SCOPE}"
  --pref_loss_type "${PREF_LOSS_TYPE}"
  --pref_lambda_init "${PREF_LAMBDA_INIT}"
  --pref_lambda_lr "${PREF_LAMBDA_LR}"
  --pref_lambda_max "${PREF_LAMBDA_MAX}"
  --pref_lambda_ema "${PREF_LAMBDA_EMA}"
  --pref_violation_clip "${PREF_VIOLATION_CLIP}"
  --pref_violation_target "${PREF_VIOLATION_TARGET}"
  --pref_lagrangian_violation_type "${PREF_LAGRANGIAN_VIOLATION_TYPE}"
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

if [[ "${ALPHA_UPDATE_STUDENT_ONLY}" == "1" ]]; then
  ARGS+=(--alpha_update_student_only)
fi

if [[ "${PREF_STOPGRAD_POSITIVE}" == "1" ]]; then
  ARGS+=(--pref_stopgrad_positive)
fi

if [[ "${STORE_INTERVENED_IN_DEMO_BUFFER}" == "1" ]]; then
  ARGS+=(--store_intervened_in_demo_buffer)
fi

if [[ "${PVP_INCLUDE_ENV_REWARD_IN_TD}" == "1" ]]; then
  ARGS+=(--pvp_include_env_reward_in_td)
fi

if [[ "${EXPORT_REPLAY_DATASET_INTERVAL}" != "0" ]]; then
  ARGS+=(--export_replay_dataset_interval "${EXPORT_REPLAY_DATASET_INTERVAL}")
fi

if [[ -n "${EXPORT_REPLAY_DATASET_PATH}" ]]; then
  ARGS+=(--export_replay_dataset_path "${EXPORT_REPLAY_DATASET_PATH}")
fi

if [[ -n "${EXPORT_REPLAY_DATASET_DIR}" ]]; then
  ARGS+=(--export_replay_dataset_dir "${EXPORT_REPLAY_DATASET_DIR}")
fi

if [[ -n "${EXPORT_REPLAY_DATASET_LABEL}" ]]; then
  ARGS+=(--export_replay_dataset_label "${EXPORT_REPLAY_DATASET_LABEL}")
fi

if [[ "${EXPORT_REPLAY_DATASET_MAX_ROWS}" != "0" ]]; then
  ARGS+=(--export_replay_dataset_max_rows "${EXPORT_REPLAY_DATASET_MAX_ROWS}")
fi

if [[ "${PROFILE_TIMING}" == "1" ]]; then
  ARGS+=(--profile_timing)
fi

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
  "${ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
