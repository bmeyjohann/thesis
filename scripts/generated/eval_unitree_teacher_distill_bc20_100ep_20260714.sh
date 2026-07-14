#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis

CONTROLLER=policy \
MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_teacher_distill_bc20_resume2500_20260714/step_2500.pt" \
RUN_NAME=teacher_distill_bc20_effective_step_5000 \
OUTPUT_DIR="$ROOT/logs/unitree_mjlab/student_eval_100ep_20260713" \
RECORD_VIDEO=0 \
SEED=101 \
NUM_ENVS=32 \
NUM_EPISODES=100 \
EPISODE_LENGTH_S=60 \
GOAL_THROUGH_OBSTACLE_PROB=0.7 \
GOAL_DISTANCE_MIN=2.8 \
GOAL_DISTANCE_MAX=4.0 \
MIN_GOAL_OBSTACLE_CLEARANCE=0.9 \
DEBUG_TERRAIN_ROWS=5 \
DEBUG_TERRAIN_COLS=10 \
"$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"
