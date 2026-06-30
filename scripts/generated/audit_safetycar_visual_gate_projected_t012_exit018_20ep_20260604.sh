#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"

export EPISODES="${EPISODES:-20}"
export SEED="${SEED:-10065}"
export TEACHER_MODE_OVERRIDE="student_projected_clearance"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.12}"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.18}"
export SCRIPTED_GEO_SAFETY_MARGIN="${SCRIPTED_GEO_SAFETY_MARGIN:-0.28}"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="${SCRIPTED_GEO_EMERGENCY_CLEARANCE:-0.10}"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="${SCRIPTED_GEO_ACTION_SHIELD_STEPS:-0}"
export OUT_DIR="${OUT_DIR:-$ROOT/logs/safetygym_eval_audits/visual_gate_projected_t012_exit018_teacher_m028_e010_shield0_${EPISODES}ep_20260604}"

exec "$ROOT/scripts/generated/audit_safetycar_visual_gate_candidate_20260603.sh"
