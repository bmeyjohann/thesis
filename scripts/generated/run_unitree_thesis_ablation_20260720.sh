#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: $0 pref_only|bc_only [seed] [tag]}"
SEED="${2:-0}"
TAG="${3:-20260720_paired3seed}"
case "$VARIANT" in
  pref_only)
    ACTOR_BC_WEIGHT=0.0
    PREF_RANK_WEIGHT=1.0
    ;;
  bc_only)
    ACTOR_BC_WEIGHT=0.2
    PREF_RANK_WEIGHT=0.0
    ;;
  *) echo "Unsupported ablation: $VARIANT" >&2; exit 2 ;;
esac

ROOT=/home/benjamin/thesis
export REPO_ROOT="$ROOT"
export METHOD=thesis
export SEED
export RUN_NAME="unitree_ablation_${VARIANT}_seed${SEED}_${TAG}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-thesis-unitree-nav}"
export WANDB_GROUP="${WANDB_GROUP:-unitree_thesis_ablation_${TAG}}"

# Matched collection and optimization budget.
export NUM_ENVS=8
export TOTAL_STEPS=5000
export EPISODE_LENGTH_S=90
export LEARNING_STARTS=300
export RANDOM_STEPS=0
export TEACHER_WARMUP_STEPS=0
export BATCH_SIZE=256
export UPDATES_PER_STEP=4
export POLICY_FREQUENCY=2
export N_STEP=5
export CHECKPOINT_INTERVAL=500
export EVAL_INTERVAL=1000
export EVAL_NUM_ENVS=8
export EVAL_NUM_EPISODES=16
export EVAL_SEED=941
export EVAL_LAYOUT_MANIFEST="$ROOT/config/unitree_eval_manifest_blocked100_seed941.json"
export EVAL_AT_END=1

if [[ ! -f "$EVAL_LAYOUT_MANIFEST" ]]; then
  echo "Missing paired evaluation manifest: $EVAL_LAYOUT_MANIFEST" >&2
  exit 2
fi

# Identical goal-reaching initialization and policy architecture.
export INIT_ACTOR_CHECKPOINT="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt"
export HEIGHT_SCAN_RESOLUTION=0.25
export SCAN_HISTORY=1
export ACTION_HISTORY=0
export POLICY_ENCODER=mlp
export DETERMINISTIC_STUDENT=1
export ALPHA_INIT=0.01
export ALPHA_MIN=0.01
export ALPHA_MAX=0.01

# Shared teacher and intervention criterion.
export TEACHER_TYPE=geom_scan
export INTERVENTION_GATE_MODE=clearance_or_stall
export INTERVENTION_CLEARANCE_MODE=teacher_ratio
export INTERVENTION_CLEARANCE_TRIGGER_RATIO=1.25
export INTERVENTION_CLEARANCE_RELEASE_RATIO=1.50
export INTERVENTION_STALL_STEPS=120
export INTERVENTION_RELEASE_STEPS=4
export INTERVENTION_RELEASE_ACTION_DELTA_MAX=10.0
export TEACHER_GEOM_CLEARANCE=0.50
export TEACHER_GEOM_COMMAND_SMOOTHING=0.70
export TEACHER_GEOM_WAYPOINT_COMMIT_DISTANCE=0.50
export TEACHER_GEOM_WAYPOINT_REACH_DIST=0.25
export TEACHER_GOAL_STOP_DIST=0.30

# Shared learner reward; only these two auxiliary-loss weights differ.
export LEARNER_REWARD_MODE=dense_progress
export DENSE_PROGRESS_SCALE=1.0
export SUCCESS_BONUS=20.0
export FAILURE_PENALTY=-20.0
export ACTOR_BC_WEIGHT
export PREF_RANK_WEIGHT
export PREF_LOSS_TYPE=lagrangian
export PREF_LAMBDA_LR=0.01
export PREF_LAMBDA_MAX=10.0

# Exact random multi-obstacle training distribution.
export SUCCESS_DIST=0.40
export GOAL_DISTANCE_MIN=4.5
export GOAL_DISTANCE_MAX=8.0
export DEBUG_GOAL_THROUGH_OBSTACLE=1
export GOAL_THROUGH_OBSTACLE_PROB=1.0
export REQUIRE_BLOCKED_CORRIDOR=1
export BLOCKED_CORRIDOR_RADIUS=0.45
export BLOCKED_CORRIDOR_MIN_CELLS=4
export BLOCKED_GOAL_MAX_DISTANCE=8.0
export BLOCKED_GOAL_DISTANCE_SAMPLING=uniform
export DEBUG_GOAL_OBSTACLE_MIN_DIST=1.0
export DEBUG_GOAL_OBSTACLE_MAX_DIST=5.5
export DEBUG_NUM_OBSTACLES=6
export DEBUG_OBSTACLE_WIDTH_MIN=1.0
export DEBUG_OBSTACLE_WIDTH_MAX=1.4
export DEBUG_TERRAIN_ROWS=5
export DEBUG_TERRAIN_COLS=10
export DEBUG_PLATFORM_WIDTH=2.0
export STRICT_MIN_SIZE_OBSTACLES=1
export RESAMPLE_TERRAIN_TILES=1
export MIN_START_OBSTACLE_CLEARANCE=1.0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.9

printf '%s\n' \
  "Unitree thesis component ablation" \
  "variant:     $VARIANT" \
  "seed:        $SEED" \
  "run:         $RUN_NAME" \
  "pref weight: $PREF_RANK_WEIGHT" \
  "BC weight:   $ACTOR_BC_WEIGHT" \
  "eval cohort: manifest=$EVAL_LAYOUT_MANIFEST episodes=$EVAL_NUM_EPISODES"

exec "$ROOT/scripts/run_unitree_mjlab_nav_thesis_local.sh"
