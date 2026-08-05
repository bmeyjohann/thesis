#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?history_only or history_smooth}"
MODE="${2:?eval or plot}"
ROOT=/home/benjamin/thesis
case "$VARIANT" in
  history_only) RUN=unitree_scan13_hist5_actionhist4_history_only_2500_20260714 ;;
  history_smooth) RUN=unitree_scan13_hist5_actionhist4_history_smooth_2500_20260714 ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/$RUN/step_2500.pt"

case "$MODE" in
  eval)
    exec "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
      CONTROLLER=policy \
      MODEL_PATH="$MODEL_PATH" \
      NUM_ENVS=8 \
      NUM_EPISODES=10 \
      SEED=97 \
      RECORD_VIDEO=0 \
      RUN_NAME="${RUN}_eval10_seed97" \
      OUTPUT_DIR="$ROOT/logs/unitree_mjlab/action_history_ablation"
    ;;
  plot)
    exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
      CONTROLLER=policy \
      MODEL_PATH="$MODEL_PATH" \
      NUM_ROLLOUTS=6 \
      STEPS=1200 \
      SEED=97 \
      OUTPUT_DIR="$ROOT/visualizations/unitree_nav_action_history_ablation/$VARIANT"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
