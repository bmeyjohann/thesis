#!/usr/bin/env bash
set -euo pipefail
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_pcpocostguard}"
export STEPS="${STEPS:-40000}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-0.5}"
export TEACHER_PROGRESS_BAD_STEPS="${TEACHER_PROGRESS_BAD_STEPS:-2}"
export TEACHER_PROGRESS_GOOD_STEPS="${TEACHER_PROGRESS_GOOD_STEPS:-6}"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.05}"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.12}"
exec /home/benjamin/thesis/scripts/generated/run_safetycar_thesis_method_intervention_20260518.sh pcpo_cost
