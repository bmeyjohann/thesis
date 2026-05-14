#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

TEACHER_KIND="$(printf '%s' "${TEACHER_KIND:-cpo}" | tr '[:upper:]' '[:lower:]')"
GATE_KIND="$(printf '%s' "${GATE_KIND:-clearance}" | tr '[:upper:]' '[:lower:]')"
case "$TEACHER_KIND" in
  cpo)
    TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/models/safetygym_teachers/safe_rl/cpo_model_599.pt}"
    TEACHER_CFG="${TEACHER_CFG:-${ROOT}/models/safetygym_teachers/safe_rl/cpo_eval.yaml}"
    ;;
  pcpo)
    TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/models/safetygym_teachers/safe_rl/pcpo_model_599.pt}"
    TEACHER_CFG="${TEACHER_CFG:-${ROOT}/models/safetygym_teachers/safe_rl/pcpo_eval.yaml}"
    ;;
  *)
    echo "TEACHER_KIND must be cpo or pcpo, got: ${TEACHER_KIND}" >&2
    exit 2
    ;;
esac

case "$GATE_KIND" in
  clearance)
    TEACHER_OVERRIDE_MODE_VALUE="clearance"
    ;;
  progress|teacher_goal_progress)
    GATE_KIND="progress"
    TEACHER_OVERRIDE_MODE_VALUE="teacher_goal_progress"
    ;;
  *)
    echo "GATE_KIND must be clearance or progress, got: ${GATE_KIND}" >&2
    exit 2
    ;;
esac

if [[ -z "${INIT_CHECKPOINT_PATH:-}" ]]; then
  echo "INIT_CHECKPOINT_PATH is required; use the goal-only pretrain checkpoint you want to fine-tune." >&2
  exit 2
fi

TS="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-safetycar_goal1_own_${TEACHER_KIND}_teacher_${GATE_KIND}_gate_small_ln_${TS}}"

echo "Starting SafetyCar Goal1 own-method Safe-RL teacher run"
echo "run:      ${EXP_NAME}"
echo "teacher:  ${TEACHER_KIND}"
echo "ckpt:     ${TEACHER_CKPT}"
echo "config:   ${TEACHER_CFG}"
echo "student:  ${INIT_CHECKPOINT_PATH}"
echo "pref mix: ${PREF_REPLAY_SAMPLE_RATIO:-0.5}"
echo "gate:     ${TEACHER_OVERRIDE_MODE_VALUE} enter=${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.10} exit=${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.15}"

WANDB_MODE="${WANDB_MODE:-online}" \
PROJECT="${PROJECT:-thesis-safetygym}" \
WANDB_GROUP="${WANDB_GROUP:-safetycar_saferl_teacher_probe_20260511}" \
EXP_NAME="$EXP_NAME" \
ENV_NAME="${ENV_NAME:-SafetyCarGoal1-v0}" \
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}" \
LEARNING_STARTS="${LEARNING_STARTS:-0}" \
VARIANT=own \
MODULE_IMPL=custom \
USE_LAYER_NORM=1 \
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}" \
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}" \
INIT_CHECKPOINT_PATH="$INIT_CHECKPOINT_PATH" \
LOAD_ACTOR_FROM_CHECKPOINT=1 \
LOAD_CRITIC_FROM_CHECKPOINT=1 \
LOAD_CRITIC_TARGET_FROM_CHECKPOINT=1 \
LOAD_ALPHA_FROM_CHECKPOINT=1 \
LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT=0 \
REWARD_MODE="${REWARD_MODE:-dense}" \
STEP_PENALTY="${STEP_PENALTY:--0.001}" \
COST_PENALTY=0.0 \
OBS_MASK_MODE=none \
TERMINATE_ON_GOAL=1 \
RESEED_ON_EPISODE_RESET=0 \
CAR_ACTION_MODE=raw_wheels \
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}" \
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}" \
OBS_NORMALIZATION=1 \
SCALE_ACTOR_TO_ENV_BOUNDS=1 \
USE_INTERVENTION=1 \
HUMAN_INPUT_DEVICE=safe_rl \
EXPERT_CHECKPOINT_PATH="$TEACHER_CKPT" \
EXPERT_CONFIG_PATH="$TEACHER_CFG" \
EXPERT_DEVICE="${EXPERT_DEVICE:-cpu}" \
TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.10}" \
TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.15}" \
TEACHER_OVERRIDE_MODE="$TEACHER_OVERRIDE_MODE_VALUE" \
TEACHER_GOAL_PROGRESS_STEPS="${TEACHER_GOAL_PROGRESS_STEPS:-3}" \
TEACHER_GOAL_PROGRESS_EPSILON="${TEACHER_GOAL_PROGRESS_EPSILON:-0.001}" \
PREF_CAPACITY="${PREF_CAPACITY:-100000}" \
PREF_SAMPLING_MODE=linked \
PREF_SAMPLE_RATIO=0.0 \
PREF_REPLAY_SAMPLE_RATIO="${PREF_REPLAY_SAMPLE_RATIO:-0.5}" \
PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}" \
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.1}" \
PREF_LOSS_TYPE="${PREF_LOSS_TYPE:-lagrangian}" \
PREF_STOPGRAD_POSITIVE=1 \
DEMO_SAMPLE_RATIO=0.0 \
PREFILL_DEMO_EPISODES=0 \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}" \
LOG_INTERVAL="${LOG_INTERVAL:-1000}" \
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}" \
VIZ_ON_CHECKPOINT="${VIZ_ON_CHECKPOINT:-1}" \
EVAL_SAVE_EPISODE_PLOTS="${EVAL_SAVE_EPISODE_PLOTS:-1}" \
/home/benjamin/thesis/scripts/run_safetycar_minimal_human_ready_local.sh
