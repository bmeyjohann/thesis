#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*)
      export "$arg"
      ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_NAME="${BASE_NAME:-safetycar_goal1_ppo_layout_curriculum_${TIMESTAMP}}"
WANDB_GROUP="${WANDB_GROUP:-$BASE_NAME}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"
SEED="${SEED:-1}"

COMMON_ENV=(
  "ENV_NAME=SafetyCarGoal1-v0"
  "WANDB_MODE=$WANDB_MODE"
  "PROJECT=$PROJECT"
  "WANDB_GROUP=$WANDB_GROUP"
  "SEED=$SEED"
  "DEVICE=${DEVICE:-auto}"
  "NUM_ENVS=${NUM_ENVS:-16}"
  "VEC_ENV=${VEC_ENV:-subproc}"
  "PPO_N_STEPS=${PPO_N_STEPS:-512}"
  "PPO_BATCH_SIZE=${PPO_BATCH_SIZE:-1024}"
  "PPO_N_EPOCHS=${PPO_N_EPOCHS:-10}"
  "GAMMA=${GAMMA:-0.99}"
  "GAE_LAMBDA=${GAE_LAMBDA:-0.95}"
  "PPO_LEARNING_RATE=${PPO_LEARNING_RATE:-3e-4}"
  "PPO_CLIP_RANGE=${PPO_CLIP_RANGE:-0.2}"
  "PPO_ENT_COEF=${PPO_ENT_COEF:-0.0}"
  "PPO_NET_ARCH=${PPO_NET_ARCH:-256,256}"
  "PPO_ACTIVATION_FN=${PPO_ACTIVATION_FN:-tanh}"
  "SURFACE_MODE=${SURFACE_MODE:-default}"
  "CAR_ACTION_MODE=${CAR_ACTION_MODE:-raw_wheels}"
  "CAR_WHEEL_COMMAND_LIMIT=${CAR_WHEEL_COMMAND_LIMIT:-1.0}"
  "CAR_FORCE_SCALE=${CAR_FORCE_SCALE:-1.0}"
  "REWARD_MODE=${REWARD_MODE:-potential_diff}"
  "DENSE_REWARD_SCALE=${DENSE_REWARD_SCALE:-1.0}"
  "SUCCESS_REWARD_SCALE=${SUCCESS_REWARD_SCALE:-10.0}"
  "STEP_PENALTY=${STEP_PENALTY:-0.0}"
  "COST_PENALTY=${COST_PENALTY:-0.0}"
  "CLEARANCE_PENALTY_MODE=${CLEARANCE_PENALTY_MODE:-softplus}"
  "CLEARANCE_MARGIN=${CLEARANCE_MARGIN:-0.05}"
  "CLEARANCE_PENALTY_SCALE=${CLEARANCE_PENALTY_SCALE:-1.1}"
  "CLEARANCE_PENALTY_TEMPERATURE=${CLEARANCE_PENALTY_TEMPERATURE:-0.001}"
  "TERMINATE_ON_GOAL=${TERMINATE_ON_GOAL:-1}"
  "TERMINATE_ON_COST=${TERMINATE_ON_COST:-1}"
  "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-50000}"
  "LOG_INTERVAL=${LOG_INTERVAL:-4096}"
  "EVAL_INTERVAL=${EVAL_INTERVAL:-50000}"
  "EVAL_EPISODES=${EVAL_EPISODES:-12}"
  "EVAL_SAVE_PLOTS=1"
  "EVAL_REWARD_SURFACE=1"
  "EVAL_PLOT_MAX_EPISODES=${EVAL_PLOT_MAX_EPISODES:-6}"
)

run_phase() {
  local phase="$1"
  local steps="$2"
  local fixed_layout="$3"
  local layout_curriculum="$4"
  local layout_level="$5"
  local init_ckpt="$6"

  local exp_name="${BASE_NAME}_${phase}"
  echo "Starting PPO curriculum phase: $exp_name"
  echo "  steps:      $steps"
  echo "  fixed:      $fixed_layout"
  echo "  curriculum: $layout_curriculum level=$layout_level"
  echo "  init:       ${init_ckpt:-<scratch>}"

  ./scripts/run_safetypoint_ppo_local.sh \
    "${COMMON_ENV[@]}" \
    "EXP_PREFIX=$exp_name" \
    "EXP_NAME=$exp_name" \
    "TOTAL_TIMESTEPS=$steps" \
    "FIXED_LAYOUT_PRESET=$fixed_layout" \
    "LAYOUT_CURRICULUM=$layout_curriculum" \
    "LAYOUT_CURRICULUM_LEVEL=$layout_level" \
    "INIT_PPO_MODEL_PATH=$init_ckpt"
}

start_ckpt="${INIT_PPO_MODEL_PATH:-$ROOT/models/safetygym_ppo/safetycar_goal1_ppo_raw_softplus_centerblock_margin005_200k_20260514/final.zip}"
if [[ ! -f "$start_ckpt" ]]; then
  echo "Start checkpoint not found, falling back to scratch: $start_ckpt"
  start_ckpt=""
fi

run_phase "p1_yaw_agent_jitter" "${P1_STEPS:-150000}" "none" "car_block_progression" "2" "$start_ckpt"
ckpt="$ROOT/models/safetygym_ppo/${BASE_NAME}_p1_yaw_agent_jitter/final.zip"

run_phase "p2_goal_obstacle_jitter" "${P2_STEPS:-200000}" "none" "car_block_progression" "3" "$ckpt"
ckpt="$ROOT/models/safetygym_ppo/${BASE_NAME}_p2_goal_obstacle_jitter/final.zip"

run_phase "p3_slalom" "${P3_STEPS:-250000}" "none" "car_block_progression" "4" "$ckpt"
ckpt="$ROOT/models/safetygym_ppo/${BASE_NAME}_p3_slalom/final.zip"

run_phase "p4_random_blocking" "${P4_STEPS:-300000}" "none" "car_block_progression" "5" "$ckpt"
ckpt="$ROOT/models/safetygym_ppo/${BASE_NAME}_p4_random_blocking/final.zip"

run_phase "p5_unfiltered_goal1" "${P5_STEPS:-500000}" "none" "none" "0" "$ckpt"

echo "PPO layout curriculum complete."
echo "Final checkpoint:"
echo "  $ROOT/models/safetygym_ppo/${BASE_NAME}_p5_unfiltered_goal1/final.zip"
echo "Periodic eval artifacts:"
echo "  $ROOT/logs/safetygym_ppo/${BASE_NAME}_*/periodic_eval"
