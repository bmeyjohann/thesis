#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

LABEL="${1:?label required}"
ROLLOUT_ID="${2:?rollout id required}"
POLICY_PATH="${3:?policy path required}"
MAX_VX="${4:?max vx required}"
shift 4
SEED=$((200 + 10#${ROLLOUT_ID}))
OUTPUT_DIR="/home/benjamin/thesis/visualizations/unitree_nav_paired_policy_20260712/${LABEL}/rollout_${ROLLOUT_ID}"

/home/benjamin/thesis/scripts/generated/run_unitree_geom_scan_teacher_plot.sh \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --low-level-policy-path "${POLICY_PATH}" \
  --layout-generation-attempts 10 \
  --steps 600 \
  --episode-length-s 30.0 \
  --start-clearance-resample-attempts 1 \
  --blocked-corridor-resample-attempts 1 \
  --teacher-max-vx "${MAX_VX}" \
  --teacher-goal-stop-dist 0.40 \
  --teacher-geom-clearance 0.60 \
  --teacher-geom-side-penalty 8.0 \
  --teacher-geom-disengage-clear-steps 20 \
  "$@"
