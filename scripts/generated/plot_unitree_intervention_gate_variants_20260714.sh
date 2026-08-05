#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?current, balanced, or obstacle_only}"
case "$VARIANT" in
  current)
    CLEARANCE=0.90; RELEASE_CLEARANCE=1.05; STALL_STEPS=30; RELEASE_STEPS=8; RELEASE_DELTA=0.35 ;;
  balanced)
    CLEARANCE=0.75; RELEASE_CLEARANCE=0.85; STALL_STEPS=60; RELEASE_STEPS=4; RELEASE_DELTA=0.80 ;;
  obstacle_only)
    CLEARANCE=0.75; RELEASE_CLEARANCE=0.85; STALL_STEPS=100000; RELEASE_STEPS=4; RELEASE_DELTA=0.80 ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

ROOT=/home/benjamin/thesis
MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_scan13_hist5_actionhist4_history_only_2500_20260714/step_2500.pt"
exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
  CONTROLLER=policy \
  MODEL_PATH="$MODEL" \
  POLICY_TEACHER_GATE=1 \
  INTERVENTION_CLEARANCE_THRESHOLD="$CLEARANCE" \
  INTERVENTION_RELEASE_CLEARANCE="$RELEASE_CLEARANCE" \
  INTERVENTION_STALL_STEPS="$STALL_STEPS" \
  INTERVENTION_RELEASE_STEPS="$RELEASE_STEPS" \
  INTERVENTION_RELEASE_ACTION_DELTA_MAX="$RELEASE_DELTA" \
  TEACHER_GEOM_CLEARANCE=0.60 \
  TEACHER_GOAL_STOP_DIST=0.20 \
  NUM_ROLLOUTS=4 \
  STEPS=1200 \
  SEED=109 \
  OUTPUT_DIR="$ROOT/visualizations/unitree_nav_gate_diagnosis/$VARIANT"
