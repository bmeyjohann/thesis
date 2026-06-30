#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

# Retrain the thesis method from the goal-only baseline under the visual-footprint
# safety semantics. Keep learner reward clean: dense Euclidean goal progress only.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m028_t045_exit050_directional_e012_shield0}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-106}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260603}"

export INIT_CKPT="${INIT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"
export LOAD_CRITIC_FROM_CHECKPOINT="${LOAD_CRITIC_FROM_CHECKPOINT:-0}"
export LOAD_CRITIC_TARGET_FROM_CHECKPOINT="${LOAD_CRITIC_TARGET_FROM_CHECKPOINT:-0}"
export LOAD_ALPHA_FROM_CHECKPOINT="${LOAD_ALPHA_FROM_CHECKPOINT:-0}"
export LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT="${LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT:-0}"

export REWARD_MODE="dense"
export DENSE_REWARD_SCALE="1.0"
export SUCCESS_REWARD_SCALE="0.0"
export STEP_PENALTY="0.0"
export CLEARANCE_PENALTY_SCALE="0.0"

export FOOTPRINT_COST="1"
export FOOTPRINT_COST_MODE="visual"
export FOOTPRINT_COST_MARGIN="0.0"
export FOOTPRINT_COST_VALUE="1.0"

export TEACHER_MODE_OVERRIDE="clearance"
export TEACHER_CLEARANCE_SOURCE="visual_footprint"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="0.45"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="0.50"

export SCRIPTED_GEO_HEADING_TOLERANCE="${SCRIPTED_GEO_HEADING_TOLERANCE:-0.20}"
export SCRIPTED_GEO_LOOKAHEAD="1.6"
export SCRIPTED_GEO_SAFETY_MARGIN="0.28"
export SCRIPTED_GEO_GRID_RESOLUTION="0.05"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="0.12"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="0"

# Best stable single-frame thesis-method family: strong linked preference plus
# BC on intervention rows, fixed alpha, bounded lambda, and action-delta weights.
export OBS_FRAME_STACK="1"
export TEMPORAL_ENCODER="none"
export ALPHA_INIT="${ALPHA_INIT:-0.001}"
export ALPHA_MIN="${ALPHA_MIN:-0.001}"
export ALPHA_MAX="${ALPHA_MAX:-0.001}"
export PREF_STOPGRAD_POSITIVE="0"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-1.0}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.00025}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-3.0}"
export PREF_ACTION_DELTA_MIN="${PREF_ACTION_DELTA_MIN:-0.25}"
export PREF_ACTION_DELTA_WEIGHT_SCALE="${PREF_ACTION_DELTA_WEIGHT_SCALE:-1.0}"
export PREF_ACTION_DELTA_WEIGHT_MAX="${PREF_ACTION_DELTA_WEIGHT_MAX:-4.0}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-1.0}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-3.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-3.0}"

export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"
export DEMO_SAMPLE_RATIO="0.0"
export DEMO_PRETRAIN_UPDATES="0"
export DISABLE_POLICY_VIZ="${DISABLE_POLICY_VIZ:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"

exec "$BASE"
