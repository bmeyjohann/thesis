#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_cube_single_task1_relonly_component_variant_local.sh [options]

Options:
  --method own|hilserl
  --num-critics N
  --num-updates N
  --cta-ratio N
  --gamma FLOAT
  --demo-sample-ratio FLOAT
  --intervention-episode-prob FLOAT
  --pref-sampling-mode linked|independent
  --pref-sample-ratio FLOAT
  --pref-loss-type lagrangian|margin|bradley_terry
  --pref-rank-weight FLOAT
  --pref-rank-margin FLOAT
  --pref-lambda-init FLOAT
  --pref-lambda-lr FLOAT
  --pref-lambda-max FLOAT
  --pref-lambda-ema FLOAT
  --pref-violation-clip FLOAT
  --pref-violation-target FLOAT
  --pref-lagrangian-violation-type hinge|smooth
  --pref-stopgrad-positive
  --no-pref-stopgrad-positive
  --use-layer-norm
  --no-layer-norm
  --disable-rotation
  --enable-rotation
  --tolerance-xyz FLOAT
  --tolerance-yaw FLOAT
  --tolerance-gripper FLOAT
  --tolerance-near-distance FLOAT
  --tolerance-far-distance FLOAT
  --tolerance-near-scale FLOAT
  --total-timesteps N
  --num-envs N
  --batch-size N
  --learning-starts N
  --eval-interval N
  --num-eval-episodes N
  --eval-num-envs N
  --save-interval N
  --log-interval N
  --exp-name NAME
  --name-suffix TEXT
  --help
EOF
}

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"

METHOD="own"
NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
NUM_UPDATES="${NUM_UPDATES:-1}"
CTA_RATIO="${CTA_RATIO:-2}"
GAMMA="${GAMMA:-0.97}"

DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-20}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-1.0}"

USE_LAYER_NORM=1
DISABLE_ROTATION=0

EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"

TOLERANCE_TYPE="component"
TOLERANCE_VALUE="${TOLERANCE_VALUE:-30.0}"
TOL_XYZ="${TOL_XYZ:-0.35}"
TOL_YAW="${TOL_YAW:-0.45}"
TOL_GRIPPER="${TOL_GRIPPER:-0.90}"
TOL_NEAR_DIST="${TOL_NEAR_DIST:-0.08}"
TOL_FAR_DIST="${TOL_FAR_DIST:-0.30}"
TOL_NEAR_SCALE="${TOL_NEAR_SCALE:-0.35}"
TOL_YAW_SET=0

PREF_SAMPLING_MODE="linked"
PREF_SAMPLE_RATIO="0.0"
PREF_SAMPLE_RATIO_SET=0
PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.01}"
PREF_LOSS_TYPE="lagrangian"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-1e-3}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"
PREF_VIOLATION_TARGET="${PREF_VIOLATION_TARGET:-0.0}"
PREF_LAGRANGIAN_VIOLATION_TYPE="hinge"
PREF_STOPGRAD_POSITIVE=1

EXP_NAME=""
NAME_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --num-critics)
      NUM_CRITICS="$2"
      shift 2
      ;;
    --num-updates)
      NUM_UPDATES="$2"
      shift 2
      ;;
    --cta-ratio)
      CTA_RATIO="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --demo-sample-ratio)
      DEMO_SAMPLE_RATIO="$2"
      shift 2
      ;;
    --intervention-episode-prob)
      INTERVENTION_EPISODE_PROB="$2"
      shift 2
      ;;
    --pref-sampling-mode)
      PREF_SAMPLING_MODE="$2"
      shift 2
      ;;
    --pref-sample-ratio)
      PREF_SAMPLE_RATIO="$2"
      PREF_SAMPLE_RATIO_SET=1
      shift 2
      ;;
    --pref-loss-type)
      PREF_LOSS_TYPE="$2"
      shift 2
      ;;
    --pref-rank-weight)
      PREF_RANK_WEIGHT="$2"
      shift 2
      ;;
    --pref-rank-margin)
      PREF_RANK_MARGIN="$2"
      shift 2
      ;;
    --pref-lambda-init)
      PREF_LAMBDA_INIT="$2"
      shift 2
      ;;
    --pref-lambda-lr)
      PREF_LAMBDA_LR="$2"
      shift 2
      ;;
    --pref-lambda-max)
      PREF_LAMBDA_MAX="$2"
      shift 2
      ;;
    --pref-lambda-ema)
      PREF_LAMBDA_EMA="$2"
      shift 2
      ;;
    --pref-violation-clip)
      PREF_VIOLATION_CLIP="$2"
      shift 2
      ;;
    --pref-violation-target)
      PREF_VIOLATION_TARGET="$2"
      shift 2
      ;;
    --pref-lagrangian-violation-type)
      PREF_LAGRANGIAN_VIOLATION_TYPE="$2"
      shift 2
      ;;
    --pref-stopgrad-positive)
      PREF_STOPGRAD_POSITIVE=1
      shift
      ;;
    --no-pref-stopgrad-positive)
      PREF_STOPGRAD_POSITIVE=0
      shift
      ;;
    --use-layer-norm)
      USE_LAYER_NORM=1
      shift
      ;;
    --no-layer-norm)
      USE_LAYER_NORM=0
      shift
      ;;
    --disable-rotation)
      DISABLE_ROTATION=1
      shift
      ;;
    --enable-rotation)
      DISABLE_ROTATION=0
      shift
      ;;
    --tolerance-xyz)
      TOL_XYZ="$2"
      shift 2
      ;;
    --tolerance-yaw)
      TOL_YAW="$2"
      TOL_YAW_SET=1
      shift 2
      ;;
    --tolerance-gripper)
      TOL_GRIPPER="$2"
      shift 2
      ;;
    --tolerance-near-distance)
      TOL_NEAR_DIST="$2"
      shift 2
      ;;
    --tolerance-far-distance)
      TOL_FAR_DIST="$2"
      shift 2
      ;;
    --tolerance-near-scale)
      TOL_NEAR_SCALE="$2"
      shift 2
      ;;
    --total-timesteps)
      TOTAL_TIMESTEPS="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning-starts)
      LEARNING_STARTS="$2"
      shift 2
      ;;
    --eval-interval)
      EVAL_INTERVAL="$2"
      shift 2
      ;;
    --num-eval-episodes)
      NUM_EVAL_EPISODES="$2"
      shift 2
      ;;
    --eval-num-envs)
      EVAL_NUM_ENVS="$2"
      shift 2
      ;;
    --save-interval)
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --log-interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    --name-suffix)
      NAME_SUFFIX="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${METHOD}" != "own" && "${METHOD}" != "hilserl" ]]; then
  echo "--method must be one of: own, hilserl" >&2
  exit 2
fi

if [[ "${DISABLE_ROTATION}" == "1" && "${TOL_YAW_SET}" == "0" ]]; then
  TOL_YAW="100000.0"
fi

if [[ "${METHOD}" == "own" && "${PREF_SAMPLING_MODE}" == "independent" && "${PREF_SAMPLE_RATIO_SET}" == "0" ]]; then
  PREF_SAMPLE_RATIO="0.5"
fi

TS="$(date +%Y%m%d_%H%M%S)"
ROT_TAG="rot"
if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  ROT_TAG="norot"
fi
LN_TAG="ln"
if [[ "${USE_LAYER_NORM}" != "1" ]]; then
  LN_TAG="noln"
fi

if [[ "${METHOD}" == "own" ]]; then
  STOPGRAD_TAG="sgpos"
  if [[ "${PREF_STOPGRAD_POSITIVE}" != "1" ]]; then
    STOPGRAD_TAG="nosgpos"
  fi
  CORE_NAME="cube_single_task1_relcomp_${METHOD}_${ROT_TAG}_${LN_TAG}_c${NUM_CRITICS}_utd${NUM_UPDATES}_cta${CTA_RATIO}_${PREF_SAMPLING_MODE}_${PREF_LOSS_TYPE}_${STOPGRAD_TAG}"
else
  CORE_NAME="cube_single_task1_relcomp_${METHOD}_${ROT_TAG}_${LN_TAG}_c${NUM_CRITICS}_utd${NUM_UPDATES}_cta${CTA_RATIO}"
fi
if [[ -n "${NAME_SUFFIX}" ]]; then
  CORE_NAME="${CORE_NAME}_${NAME_SUFFIX}"
fi
EXP_NAME="${EXP_NAME:-${CORE_NAME}_${TS}}"

EFFECTIVE_WANDB_MODE="${WANDB_MODE_VALUE}"
if [[ "${EFFECTIVE_WANDB_MODE}" == "online" ]]; then
  HAS_WANDB_AUTH=0
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    HAS_WANDB_AUTH=1
  elif [[ -f "${HOME}/.netrc" ]] && grep -q "machine api.wandb.ai" "${HOME}/.netrc"; then
    HAS_WANDB_AUTH=1
  fi
  if [[ "${HAS_WANDB_AUTH}" != "1" ]]; then
    echo "[WandB] No API key configured; falling back to offline mode for this run."
    EFFECTIVE_WANDB_MODE="offline"
  fi
fi

LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "Method: ${METHOD}"
echo "WANDB mode: ${EFFECTIVE_WANDB_MODE}"
echo "Rotation mode: ${ROT_TAG}"
echo "LayerNorm: ${USE_LAYER_NORM}"
echo "Critics: ${NUM_CRITICS}"
echo "UTD (num_updates): ${NUM_UPDATES}"
echo "CTA ratio: ${CTA_RATIO}"
echo "Intervention episode probability: ${INTERVENTION_EPISODE_PROB}"
echo "Tolerance mode: ${TOLERANCE_TYPE} xyz=${TOL_XYZ} yaw=${TOL_YAW} gripper=${TOL_GRIPPER}"
if [[ "${METHOD}" == "own" ]]; then
  echo "Preference sampling: ${PREF_SAMPLING_MODE} pref_sample_ratio=${PREF_SAMPLE_RATIO}"
  echo "Preference loss: ${PREF_LOSS_TYPE} stopgrad_positive=${PREF_STOPGRAD_POSITIVE}"
else
  echo "Preference loss: disabled (HILSERL-style demo duplication)"
fi

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --num_envs "${NUM_ENVS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --device auto
  --train_render_mode none
  --obs_mode state
  --no_include_goal
  --include_relative_cube_features
  --relative_only_obs
  --reward_type sparse
  --cube_reward_mode dense
  --use_intervention
  --intervention_mode agent
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
  --alpha_min 0.0
  --alpha_max 1.0
  --alpha_freeze_steps 0
  --demo_buffer_enable
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
  --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
  --demo_prefill_target demo
  --demo_prefill_intervention_mode agent_always
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
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
  --tolerance_type "${TOLERANCE_TYPE}"
  --tolerance_value "${TOLERANCE_VALUE}"
  --tolerance_xyz_value "${TOL_XYZ}"
  --tolerance_yaw_value "${TOL_YAW}"
  --tolerance_gripper_value "${TOL_GRIPPER}"
  --tolerance_adaptive_enable
  --tolerance_adaptive_near_distance "${TOL_NEAR_DIST}"
  --tolerance_adaptive_far_distance "${TOL_FAR_DIST}"
  --tolerance_adaptive_near_scale "${TOL_NEAR_SCALE}"
)

if [[ "${USE_LAYER_NORM}" == "1" ]]; then
  COMMON_ARGS+=(--use_layer_norm)
fi

if [[ "${DISABLE_ROTATION}" == "1" ]]; then
  COMMON_ARGS+=(--disable_rotation)
fi

METHOD_ARGS=()
if [[ "${METHOD}" == "own" ]]; then
  METHOD_ARGS+=(
    --pref_buffer_enable
    --pref_sampling_mode "${PREF_SAMPLING_MODE}"
    --pref_sample_ratio "${PREF_SAMPLE_RATIO}"
    --pref_rank_weight "${PREF_RANK_WEIGHT}"
    --pref_rank_margin "${PREF_RANK_MARGIN}"
    --pref_loss_type "${PREF_LOSS_TYPE}"
    --pref_lambda_init "${PREF_LAMBDA_INIT}"
    --pref_lambda_lr "${PREF_LAMBDA_LR}"
    --pref_lambda_max "${PREF_LAMBDA_MAX}"
    --pref_lambda_ema "${PREF_LAMBDA_EMA}"
    --pref_violation_clip "${PREF_VIOLATION_CLIP}"
    --pref_violation_target "${PREF_VIOLATION_TARGET}"
    --pref_lagrangian_violation_type "${PREF_LAGRANGIAN_VIOLATION_TYPE}"
  )
  if [[ "${PREF_STOPGRAD_POSITIVE}" == "1" ]]; then
    METHOD_ARGS+=(--pref_stopgrad_positive)
  fi
else
  METHOD_ARGS+=(
    --store_intervened_in_demo_buffer
    --pref_sample_ratio 0.0
    --pref_rank_weight 0.0
  )
fi

WANDB_MODE="${EFFECTIVE_WANDB_MODE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
  "${COMMON_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
