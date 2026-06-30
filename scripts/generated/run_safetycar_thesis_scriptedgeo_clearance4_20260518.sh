#!/usr/bin/env bash
set -euo pipefail

export STEPS="${STEPS:-40000}"
export CLEARANCE_PENALTY_SCALE="${CLEARANCE_PENALTY_SCALE:-4.0}"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance4}"

exec /home/benjamin/thesis/scripts/generated/run_safetycar_thesis_method_intervention_20260518.sh scriptedgeo_reward
