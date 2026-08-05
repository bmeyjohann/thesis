#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-history_only}"
case "$VARIANT" in
  history_only) SMOOTHING=0.0 ;;
  history_smooth) SMOOTHING=0.6 ;;
  *) echo "Expected history_only or history_smooth" >&2; exit 2 ;;
esac

export RUN_NAME="unitree_scan13_hist5_actionhist4_${VARIANT}_2500_20260714"
export WANDB_MODE=online
export WANDB_PROJECT=thesis-unitree-nav
export WANDB_GROUP=unitree_action_history_oscillation_20260714
export SEED=41
export NUM_ENVS=8
export TOTAL_STEPS=2500
export EPISODE_LENGTH_S=60
export LEARNING_STARTS=750
export RANDOM_STEPS=300
export TEACHER_WARMUP_STEPS=750
export BATCH_SIZE=256
export UPDATES_PER_STEP=1
export POLICY_FREQUENCY=2
export HIDDEN_DIM=256
export POLICY_ENCODER=scan_cnn
export HEIGHT_SCAN_RESOLUTION=0.25
export SCAN_HISTORY=5
export ACTION_HISTORY=4
export STUDENT_ACTION_SMOOTHING="$SMOOTHING"
export GAMMA=0.99
export ALPHA_INIT=0.001
export ALPHA_MIN=0.0001
export ALPHA_MAX=0.01
export ACTOR_BC_WEIGHT=0.2
export PREF_RANK_WEIGHT=1.0
export PREF_RANK_MARGIN=0.05
export PREF_LOSS_TYPE=lagrangian
export PREF_LAMBDA_LR=0.01
export PREF_LAMBDA_MAX=10.0
export PREF_ACTION_DELTA_MIN=0.05
export PREF_STOPGRAD_POSITIVE=1
export LEARNER_REWARD_MODE=dense_progress
export DENSE_PROGRESS_SCALE=1.0
export SUCCESS_BONUS=2.0
export TEACHER_TYPE=geom_scan
export TEACHER_GEOM_CLEARANCE=0.60
export TEACHER_GOAL_STOP_DIST=0.20
export INTERVENTION_GATE_MODE=clearance_or_stall
export INTERVENTION_CLEARANCE_THRESHOLD=0.90
export INTERVENTION_RELEASE_CLEARANCE=1.05
export INTERVENTION_STALL_STEPS=30
export INTERVENTION_PROGRESS_EPSILON=0.04
export INTERVENTION_RELEASE_STEPS=8
export INTERVENTION_RELEASE_ACTION_DELTA_MAX=0.35
export SUCCESS_DIST=0.25
export GOAL_THROUGH_OBSTACLE_PROB=0.7
export GOAL_DISTANCE_MIN=2.8
export GOAL_DISTANCE_MAX=4.0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.9
export DEBUG_OBSTACLE_WIDTH_MIN=1.0
export DEBUG_OBSTACLE_WIDTH_MAX=1.4
export DEBUG_OBSTACLE_HEIGHT_MIN=1.0
export DEBUG_OBSTACLE_HEIGHT_MAX=1.0
export DEBUG_NUM_OBSTACLES=6
export DEBUG_PLATFORM_WIDTH=2.0
export DEBUG_TERRAIN_ROWS=5
export DEBUG_TERRAIN_COLS=10
export LOG_INTERVAL=100
export CHECKPOINT_INTERVAL=1250
export LOW_LEVEL_POLICY_PATH=/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_nav_thesis_local.sh
