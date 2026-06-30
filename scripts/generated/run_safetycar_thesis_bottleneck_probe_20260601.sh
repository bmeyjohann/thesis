#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required: privileged_rich_mlp|privileged_rich_framestack4|framestack4_aug_small|framestack4_aug_medium|framestack4_aug_large|framestack4_attention_short}"
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh"

export STEPS="${STEPS:-60000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_bottleneck_probe_20260601}"
export WANDB_MODE="${WANDB_MODE:-online}"

case "$VARIANT" in
  privileged_rich_mlp)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_privileged_rich_mlp_bcstrong_fixedalpha_60k}"
    export SEED="${SEED:-71}"
    export OBS_MASK_MODE="privileged_geometry_rich"
    export OBS_FRAME_STACK="1"
    export TEMPORAL_ENCODER="none"
    exec "$BASE" hybrid_bcstrong_fixedalpha_scratch
    ;;
  privileged_rich_framestack4)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_privileged_rich_framestack4_bcstrong_fixedalpha_60k}"
    export SEED="${SEED:-72}"
    export OBS_MASK_MODE="privileged_geometry_rich"
    export OBS_FRAME_STACK="4"
    export TEMPORAL_ENCODER="none"
    exec "$BASE" framestack_bcstrong_fixedalpha_scratch
    ;;
  framestack4_aug_small)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_augsmall_bcstrong_fixedalpha_60k}"
    export SEED="${SEED:-73}"
    export PREF_ACTION_NOISE_COPIES="2"
    export PREF_ACTION_NOISE_STD="0.01"
    export PREF_ACTION_DELTA_MIN="0.15"
    export PREF_ACTION_DELTA_WEIGHT_SCALE="0.5"
    export PREF_ACTION_DELTA_WEIGHT_MAX="3.0"
    exec "$BASE" framestack_bcstrong_fixedalpha_scratch
    ;;
  framestack4_aug_medium)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_augmedium_bcstrong_fixedalpha_60k}"
    export SEED="${SEED:-74}"
    export PREF_ACTION_NOISE_COPIES="3"
    export PREF_ACTION_NOISE_STD="0.025"
    export PREF_ACTION_DELTA_MIN="0.20"
    export PREF_ACTION_DELTA_WEIGHT_SCALE="1.0"
    export PREF_ACTION_DELTA_WEIGHT_MAX="4.0"
    exec "$BASE" framestack_bcstrong_fixedalpha_scratch
    ;;
  framestack4_aug_large)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_auglarge_bcstrong_fixedalpha_60k}"
    export SEED="${SEED:-75}"
    export PREF_ACTION_NOISE_COPIES="4"
    export PREF_ACTION_NOISE_STD="0.075"
    export PREF_ACTION_DELTA_MIN="0.25"
    export PREF_ACTION_DELTA_WEIGHT_SCALE="1.0"
    export PREF_ACTION_DELTA_WEIGHT_MAX="4.0"
    exec "$BASE" framestack_bcstrong_fixedalpha_scratch
    ;;
  framestack4_attention_short)
    export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_attention_bcstrong_fixedalpha_probe_60k}"
    export SEED="${SEED:-76}"
    export OBS_FRAME_STACK="4"
    export TEMPORAL_ENCODER="attention"
    exec "$BASE" framestack_attention_hybrid_scratch
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac
