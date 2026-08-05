#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?user_ratio, conservative_ratio, or terrain_only}"
case "$VARIANT" in
  user_ratio)
    TRIGGER_RATIO=0.6666666667
    RELEASE_RATIO=0.8333333333
    POLICY_TEACHER_GATE=1
    ;;
  conservative_ratio)
    TRIGGER_RATIO=1.25
    RELEASE_RATIO=1.4166666667
    POLICY_TEACHER_GATE=1
    ;;
  terrain_only)
    TRIGGER_RATIO=0.6666666667
    RELEASE_RATIO=0.8333333333
    POLICY_TEACHER_GATE=0
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

ROOT=/home/benjamin/thesis
MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_scan13_hist5_actionhist4_history_only_2500_20260714/step_2500.pt"
exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
  CONTROLLER=policy \
  MODEL_PATH="$MODEL" \
  POLICY_TEACHER_GATE="$POLICY_TEACHER_GATE" \
  STRICT_MIN_SIZE_OBSTACLES=1 \
  INTERVENTION_CLEARANCE_MODE=teacher_ratio \
  INTERVENTION_CLEARANCE_TRIGGER_RATIO="$TRIGGER_RATIO" \
  INTERVENTION_CLEARANCE_RELEASE_RATIO="$RELEASE_RATIO" \
  INTERVENTION_STALL_STEPS=60 \
  INTERVENTION_RELEASE_STEPS=4 \
  INTERVENTION_RELEASE_ACTION_DELTA_MAX=0.8 \
  TEACHER_GEOM_CLEARANCE=0.60 \
  TEACHER_GOAL_STOP_DIST=0.20 \
  NUM_ROLLOUTS=4 \
  STEPS=1200 \
  SEED=151 \
  OUTPUT_DIR="$ROOT/visualizations/unitree_nav_adaptive_gate_strict_obstacles/$VARIANT"
