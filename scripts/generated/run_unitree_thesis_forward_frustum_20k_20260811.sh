#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

RUN_NAME="${RUN_NAME:-unitree_thesis_pref_rl_frustum17_20k_$(date +%Y%m%d_%H%M%S)}"

exec ./scripts/run_unitree_mjlab_nav_thesis_local.sh \
  RUN_NAME="$RUN_NAME" \
  METHOD=thesis \
  ACTOR_BC_WEIGHT=0.0 \
  TOTAL_STEPS=20000 \
  NUM_ENVS=32 \
  N_STEP=5 \
  POLICY_ENCODER=scan_cnn \
  SCAN_HISTORY=5 \
  ACTION_HISTORY=4 \
  HEIGHT_SCAN_PATTERN=forward_frustum \
  HEIGHT_SCAN_FRUSTUM_NEAR=0.25 \
  HEIGHT_SCAN_FRUSTUM_FAR=4.0 \
  HEIGHT_SCAN_FRUSTUM_FOV_DEG=70.0 \
  HEIGHT_SCAN_FRUSTUM_SIDE=17 \
  SUCCESS_DIST=0.4 \
  REQUIRE_BLOCKED_CORRIDOR=1 \
  RESAMPLE_TERRAIN_TILES=1 \
  TEACHER_TYPE=geom_scan \
  INTERVENTION_GATE_MODE=clearance_or_stall \
  EVAL_INTERVAL=2500 \
  EVAL_NUM_ENVS=8 \
  EVAL_NUM_EPISODES=32 \
  CHECKPOINT_INTERVAL=2500 \
  WANDB_GROUP=unitree_forward_frustum_thesis
