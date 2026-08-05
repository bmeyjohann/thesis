#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_scan13_hist5_actionhist4_adaptive_safe_strict_20k_20260714/step_2500.pt"
for _ in $(seq 1 240); do
  [[ -f "$MODEL" ]] && break
  sleep 15
done
if [[ ! -f "$MODEL" ]]; then
  echo "Timed out waiting for $MODEL" >&2
  exit 1
fi

exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
  CONTROLLER=policy \
  MODEL_PATH="$MODEL" \
  POLICY_TEACHER_GATE=0 \
  NUM_ROLLOUTS=6 \
  STEPS=1200 \
  SEED=173 \
  OUTPUT_DIR="$ROOT/visualizations/unitree_nav_adaptive_strict_step2500_eval"
