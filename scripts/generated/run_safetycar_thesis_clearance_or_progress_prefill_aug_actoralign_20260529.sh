#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_prefill_aug_actoralign}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-24}"
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"
export PREF_SAMPLING_MODE="${PREF_SAMPLING_MODE:-linked}"

# Hybrid diagnostic: keep the thesis preference loss active, but add a small
# teacher-only actor loss so intervention states directly shape the policy.
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.05}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-2.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-2.0}"

exec "$BASE"
