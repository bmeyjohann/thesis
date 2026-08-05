#!/usr/bin/env bash
set -euo pipefail

export CONTROLLER=policy
export MODEL_PATH=/home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_student_scan13_history5_dense_geom_20k_20260714/step_5000.pt
export RUN_NAME=unitree_scan13_hist5_step5000_eval100_20260714
export OUTPUT_DIR=/home/benjamin/thesis/logs/unitree_mjlab/dense_scan_history_eval
export RECORD_VIDEO=0
export NUM_ENVS=16
export NUM_EPISODES=100
export EPISODE_LENGTH_S=60
export SEED=73
export SUCCESS_DIST=1.0
export HEIGHT_SCAN_RESOLUTION=0.25
export SCAN_HISTORY=5
export GOAL_DISTANCE_MIN=2.8
export GOAL_DISTANCE_MAX=4.0
export GOAL_THROUGH_OBSTACLE_PROB=0.7
export MIN_GOAL_OBSTACLE_CLEARANCE=0.9
export DEBUG_OBSTACLE_WIDTH_MIN=1.0
export DEBUG_OBSTACLE_WIDTH_MAX=1.4
export DEBUG_OBSTACLE_HEIGHT_MIN=1.0
export DEBUG_OBSTACLE_HEIGHT_MAX=1.0
export DEBUG_NUM_OBSTACLES=6
export DEBUG_PLATFORM_WIDTH=2.0
export DEBUG_TERRAIN_ROWS=5
export DEBUG_TERRAIN_COLS=10

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh
