#!/usr/bin/env bash
set -euo pipefail

export EPISODES="${EPISODES:-20}"
export OUT_DIR="${OUT_DIR:-/home/benjamin/thesis/logs/safetygym_eval_audits/visual_gate_candidate_t045_exit050_teacher_m028_directional_e012_shield0_${EPISODES}ep_20260604}"

exec /home/benjamin/thesis/scripts/generated/audit_safetycar_visual_gate_candidate_20260603.sh
