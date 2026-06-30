#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"

export EPISODES="${EPISODES:-20}"
export SEED="${SEED:-10065}"
export TEACHER_MODE_OVERRIDE="clearance_projected_release"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.50}"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.20}"
export SCRIPTED_GEO_SAFETY_MARGIN="${SCRIPTED_GEO_SAFETY_MARGIN:-0.30}"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="${SCRIPTED_GEO_EMERGENCY_CLEARANCE:-0.14}"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="${SCRIPTED_GEO_ACTION_SHIELD_STEPS:-0}"
export OUT_DIR="${OUT_DIR:-$ROOT/logs/safetygym_eval_audits/visual_gate_clearance_projected_release_t050_rel020_teacher_m030_e014_shield0_${EPISODES}ep_20260604}"

exec "$ROOT/scripts/generated/audit_safetycar_visual_gate_candidate_20260603.sh"
