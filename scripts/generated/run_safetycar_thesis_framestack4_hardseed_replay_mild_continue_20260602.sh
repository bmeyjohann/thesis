#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_framestack4_hardseed_replay_continue_20260601.sh"

export STEPS="${STEPS:-15000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_hardseed_replay_mild_20260602}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_hardseed_replay_mild_continue_15k}"
export SEED="${SEED:-87}"

# Keep hard layouts as a rehearsal signal, not the dominant online distribution.
export LAYOUT_SEED_REPLAY_PROB="${LAYOUT_SEED_REPLAY_PROB:-0.1}"
export LAYOUT_SEED_REPLAY_MODE="${LAYOUT_SEED_REPLAY_MODE:-random}"
export EVAL_LAYOUT_SEED_REPLAY_PROB="${EVAL_LAYOUT_SEED_REPLAY_PROB:-0.0}"

# Avoid the 0.5-replay failure mode where prefill + BC made the run teacher-dominated.
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-0}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-0}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.25}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-1.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-1.0}"

exec "$BASE"
