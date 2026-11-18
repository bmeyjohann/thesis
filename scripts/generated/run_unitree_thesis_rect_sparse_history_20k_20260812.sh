#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

RUN_NAME="${RUN_NAME:-unitree_thesis_pref_rl_rect_hist5_stride5_20k_$(date +%Y%m%d_%H%M%S)}"

exec ./scripts/run_unitree_mjlab_nav_thesis_local.sh \
  RUN_NAME="$RUN_NAME" \
  METHOD=thesis \
  ACTOR_BC_WEIGHT=0.0 \
  TOTAL_STEPS=20000 \
  NUM_ENVS=32 \
  N_STEP=5 \
  POLICY_ENCODER=mlp \
  HEIGHT_SCAN_PATTERN=grid \
  HEIGHT_SCAN_RESOLUTION=0.25 \
  SCAN_HISTORY=5 \
  SCAN_HISTORY_STRIDE=5 \
  ACTION_HISTORY=20 \
  SUCCESS_DIST=0.4 \
  REQUIRE_BLOCKED_CORRIDOR=1 \
  RESAMPLE_TERRAIN_TILES=1 \
  TEACHER_TYPE=geom_scan \
  INTERVENTION_GATE_MODE=clearance_or_stall \
  EVAL_INTERVAL=2500 \
  EVAL_NUM_ENVS=8 \
  EVAL_NUM_EPISODES=32 \
  CHECKPOINT_INTERVAL=2500 \
  WANDB_GROUP=unitree_rect_sparse_history_thesis
