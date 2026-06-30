#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required: fixed05|fixed025|lowpref_fixed05}"
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_method_intervention_20260518.sh"

export STEPS="${STEPS:-60000}"
export SEED="${SEED:-92}"
export OBS_FRAME_STACK="${OBS_FRAME_STACK:-1}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_flowsec_stabilized_20260602}"
export REWARD_MODE="${REWARD_MODE:-dense}"
export DENSE_REWARD_SCALE="${DENSE_REWARD_SCALE:-1.0}"
export SUCCESS_REWARD_SCALE="${SUCCESS_REWARD_SCALE:-0.0}"
export STEP_PENALTY="${STEP_PENALTY:-0.0}"
export CLEARANCE_PENALTY_SCALE="${CLEARANCE_PENALTY_SCALE:-0.0}"
export TEACHER_MODE_OVERRIDE="${TEACHER_MODE_OVERRIDE:-clearance_or_progress}"
export TEACHER_PROGRESS_SCORE_MODE="${TEACHER_PROGRESS_SCORE_MODE:-euclidean}"
export TEACHER_PROGRESS_BAD_STEPS="${TEACHER_PROGRESS_BAD_STEPS:-3}"
export TEACHER_PROGRESS_GOOD_STEPS="${TEACHER_PROGRESS_GOOD_STEPS:-5}"
export TEACHER_PROGRESS_EPSILON="${TEACHER_PROGRESS_EPSILON:-0.001}"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.08}"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.12}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export DISABLE_POLICY_VIZ="${DISABLE_POLICY_VIZ:-1}"

# Best previous structure: actor BC on intervention rows plus linked preference-Q shaping.
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-1.0}"
export ACTOR_BC_ONLY_UNTIL_STEP="${ACTOR_BC_ONLY_UNTIL_STEP:-0}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
export PREF_ACTION_DELTA_MIN="${PREF_ACTION_DELTA_MIN:-0.0}"
export PREF_STOPGRAD_POSITIVE="${PREF_STOPGRAD_POSITIVE:-1}"
export NUM_UPDATES="${NUM_UPDATES:-2}"
export POLICY_FREQUENCY="${POLICY_FREQUENCY:-2}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-5.0}"

case "$VARIANT" in
  fixed05)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_flowsec_bcpref_fixedlambda05_60k}"
    export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-0.5}"
    export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.0}"
    export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-0.5}"
    ;;
  fixed025)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_flowsec_bcpref_fixedlambda025_60k}"
    export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-0.25}"
    export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.0}"
    export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-0.25}"
    ;;
  lowpref_fixed05)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_flowsec_bcpref_lowpref_fixedlambda05_60k}"
    export PREF_SAMPLE_RATIO=0.25
    export PREF_RANK_WEIGHT=0.5
    export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-0.5}"
    export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.0}"
    export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-0.5}"
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

exec "$BASE" scriptedgeo_reward
