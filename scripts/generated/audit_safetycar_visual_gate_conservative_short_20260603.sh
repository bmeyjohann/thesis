#!/usr/bin/env bash
set -euo pipefail

export EPISODES="${EPISODES:-20}"
export PLOT_MAX_EPISODES="${PLOT_MAX_EPISODES:-12}"
export THRESHOLDS="${THRESHOLDS:-0.15 0.25 0.35}"
export SCRIPTED_GEO_HEADING_TOLERANCE="${SCRIPTED_GEO_HEADING_TOLERANCE:-0.20}"
export SCRIPTED_GEO_LOOKAHEAD="${SCRIPTED_GEO_LOOKAHEAD:-1.4}"
export SCRIPTED_GEO_SAFETY_MARGIN="${SCRIPTED_GEO_SAFETY_MARGIN:-0.35}"
export SCRIPTED_GEO_GRID_RESOLUTION="${SCRIPTED_GEO_GRID_RESOLUTION:-0.06}"
export AUDIT_ROOT="${AUDIT_ROOT:-/home/benjamin/thesis/logs/safetygym_eval_audits/visual_gate_conservative_short_20260603}"

exec /home/benjamin/thesis/scripts/generated/audit_safetycar_visual_gate_sweep_20260603.sh

