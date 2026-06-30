#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required: framestack4_actor_bc_warmup|framestack4_actor_fast}"
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh"

export STEPS="${STEPS:-60000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_actor_bottleneck_probe_20260601}"
export WANDB_MODE="${WANDB_MODE:-online}"

case "$VARIANT" in
  framestack4_actor_bc_warmup)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_actor_bc_warmup_60k}"
    export SEED="${SEED:-77}"
    export OBS_FRAME_STACK="4"
    export TEMPORAL_ENCODER="none"
    export ACTOR_BC_WEIGHT="3.0"
    export ACTOR_BC_ONLY_UNTIL_STEP="10000"
    export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="4.0"
    export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="4.0"
    exec "$BASE" framestack_bcstrong_fixedalpha_scratch
    ;;
  framestack4_actor_fast)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_actor_fast_60k}"
    export SEED="${SEED:-78}"
    export OBS_FRAME_STACK="4"
    export TEMPORAL_ENCODER="none"
    export POLICY_FREQUENCY="1"
    export ACTOR_LEARNING_RATE="0.001"
    exec "$BASE" framestack_bcstrong_fixedalpha_scratch
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac
