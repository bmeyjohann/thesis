#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_bcstrong_120k}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-33}"

# Diagnostic baseline: if teacher-row BC cannot transfer safety, the issue is
# likely representation/architecture/data coverage, not only preference tuning.
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-0.0}"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.0}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-1.0}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-2.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-2.0}"

exec "$BASE"
