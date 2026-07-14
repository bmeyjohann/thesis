#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=unitree_teacher_always_forced_clearance08_goalclear09_seed43_20260713
export WANDB_MODE=offline
export SEED=43
export NUM_ENVS=8
export TOTAL_STEPS=1200
export EPISODE_LENGTH_S=60
export LEARNING_STARTS=999999
export RANDOM_STEPS=0
export TEACHER_WARMUP_STEPS=0
export STUDENT_CONTROLLER=direct_goal
export TEACHER_TYPE=geom_scan
export INTERVENTION_GATE_MODE=always
export TEACHER_GEOM_CLEARANCE=0.80
export TEACHER_GOAL_STOP_DIST=0.40
export GOAL_THROUGH_OBSTACLE_PROB=1.0
export GOAL_DISTANCE_MIN=2.8
export GOAL_DISTANCE_MAX=4.0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.9
export DEBUG_TERRAIN_ROWS=5
export DEBUG_TERRAIN_COLS=10
export LOG_INTERVAL=100
export CHECKPOINT_INTERVAL=1250
export LOW_LEVEL_POLICY_PATH=/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_nav_thesis_local.sh
