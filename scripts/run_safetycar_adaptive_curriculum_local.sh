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
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
EASY_STEPS="${EASY_STEPS:-$((TOTAL_TIMESTEPS / 2))}"
HARD_STEPS="${HARD_STEPS:-$((TOTAL_TIMESTEPS - EASY_STEPS))}"
EASY_GOAL_STEPS="${EASY_GOAL_STEPS:-$((EASY_STEPS / 2))}"
EASY_SAFETY_STEPS="${EASY_SAFETY_STEPS:-$((EASY_STEPS - EASY_GOAL_STEPS))}"
HARD_SOFT_STEPS="${HARD_SOFT_STEPS:-$((HARD_STEPS / 2))}"
HARD_STRICT_STEPS="${HARD_STRICT_STEPS:-$((HARD_STEPS - HARD_SOFT_STEPS))}"

BASE_NAME="${BASE_NAME:-safetycar_curriculum_adaptive_${TIMESTAMP}}"
WANDB_GROUP="${WANDB_GROUP:-$BASE_NAME}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"
SEED="${SEED:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"

COMMON_ENV=(
  "WANDB_MODE=$WANDB_MODE"
  "PROJECT=$PROJECT"
  "WANDB_GROUP=$WANDB_GROUP"
  "SEED=$SEED"
  "VARIANT=plain"
  "MODULE_IMPL=custom"
  "USE_LAYER_NORM=1"
  "ACTOR_HIDDEN_DIM=${ACTOR_HIDDEN_DIM:-512}"
  "CRITIC_HIDDEN_DIM=${CRITIC_HIDDEN_DIM:-1024}"
  "REWARD_MODE=${REWARD_MODE:-dense_plus_sparse}"
  "STEP_PENALTY=${STEP_PENALTY:--0.001}"
  "NUM_UPDATES=${NUM_UPDATES:-2}"
  "POLICY_FREQUENCY=${POLICY_FREQUENCY:-2}"
  "GAMMA=${GAMMA:-0.99}"
  "CAR_WHEEL_COMMAND_LIMIT=${CAR_WHEEL_COMMAND_LIMIT:-1.0}"
  "CAR_FORCE_SCALE=${CAR_FORCE_SCALE:-1.0}"
  "CAR_ACTION_MODE=${CAR_ACTION_MODE:-raw_wheels}"
  "OBS_NORMALIZATION=${OBS_NORMALIZATION:-1}"
  "RESEED_ON_EPISODE_RESET=${RESEED_ON_EPISODE_RESET:-0}"
  "LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT=${LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT:-0}"
  "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
  "LOG_INTERVAL=${LOG_INTERVAL:-2000}"
  "NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES:-10}"
  "VIZ_ON_CHECKPOINT=${VIZ_ON_CHECKPOINT:-0}"
  "EVAL_SAVE_EPISODE_PLOTS=${EVAL_SAVE_EPISODE_PLOTS:-1}"
  "ADAPTIVE_SAFETY_CURRICULUM=1"
  "ADAPTIVE_SAFETY_GOAL_TARGET=${ADAPTIVE_SAFETY_GOAL_TARGET:-5.0}"
  "ADAPTIVE_SAFETY_WINDOW_EPISODES=${ADAPTIVE_SAFETY_WINDOW_EPISODES:-5}"
  "ADAPTIVE_SAFETY_STEP=${ADAPTIVE_SAFETY_STEP:-0.10}"
)

run_phase() {
  local phase_name="$1"
  local env_name="$2"
  local steps="$3"
  local checkpoint="$4"
  local cost_penalty="$5"
  local clearance_scale="$6"
  local clearance_margin="$7"
  local adaptive_init="$8"
  local terminate_on_goal="$9"

  if [[ "$steps" -le 0 ]]; then
    return
  fi

  local exp_name="${BASE_NAME}_${phase_name}"
  echo "Starting curriculum phase: $exp_name"
  echo "  env:        $env_name"
  echo "  steps:      $steps"
  echo "  checkpoint: ${checkpoint:-<scratch>}"
  echo "  cost:       $cost_penalty"
  echo "  clearance:  $clearance_scale @ margin $clearance_margin"
  echo "  adaptive:   init=$adaptive_init target=${ADAPTIVE_SAFETY_GOAL_TARGET:-1.0}"

  ./scripts/run_safetycar_minimal_human_ready_local.sh \
    "${COMMON_ENV[@]}" \
    "EXP_NAME=$exp_name" \
    "ENV_NAME=$env_name" \
    "TOTAL_TIMESTEPS=$steps" \
    "INIT_CHECKPOINT_PATH=$checkpoint" \
    "COST_PENALTY=$cost_penalty" \
    "CLEARANCE_PENALTY_SCALE=$clearance_scale" \
    "CLEARANCE_MARGIN=$clearance_margin" \
    "ADAPTIVE_SAFETY_INIT=$adaptive_init" \
    "TERMINATE_ON_GOAL=$terminate_on_goal"
}

phase1="${BASE_NAME}_p1_easy_goal"
phase2="${BASE_NAME}_p2_easy_safety"
phase3="${BASE_NAME}_p3_hard_soft"

run_phase "p1_easy_goal" "SafetyCarGoal1-v0" "$EASY_GOAL_STEPS" "" \
  "0.0" "0.0" "0.0" "0.0" "1"

ckpt1="$ROOT/models/safetygym_minimal/$phase1/final.pt"
run_phase "p2_easy_safety" "SafetyCarGoal1-v0" "$EASY_SAFETY_STEPS" "$ckpt1" \
  "${EASY_COST_PENALTY:--0.05}" "${EASY_CLEARANCE_PENALTY_SCALE:-0.05}" "${EASY_CLEARANCE_MARGIN:-0.15}" "0.25" "0"

ckpt2="$ROOT/models/safetygym_minimal/$phase2/final.pt"
run_phase "p3_hard_soft" "SafetyCarGoal2-v0" "$HARD_SOFT_STEPS" "$ckpt2" \
  "${HARD_SOFT_COST_PENALTY:--0.05}" "${HARD_SOFT_CLEARANCE_PENALTY_SCALE:-0.05}" "${HARD_SOFT_CLEARANCE_MARGIN:-0.15}" "0.25" "0"

ckpt3="$ROOT/models/safetygym_minimal/$phase3/final.pt"
run_phase "p4_hard_strict" "SafetyCarGoal2-v0" "$HARD_STRICT_STEPS" "$ckpt3" \
  "${HARD_STRICT_COST_PENALTY:--0.10}" "${HARD_STRICT_CLEARANCE_PENALTY_SCALE:-0.10}" "${HARD_STRICT_CLEARANCE_MARGIN:-0.20}" "0.50" "0"

echo "Curriculum complete."
echo "Final checkpoint:"
echo "  $ROOT/models/safetygym_minimal/${BASE_NAME}_p4_hard_strict/final.pt"
