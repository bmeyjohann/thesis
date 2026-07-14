#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

ROLLOUT_ID="${1:?rollout id required}"
SEED=$((100 + 10#${ROLLOUT_ID}))
OUTPUT_DIR="/home/benjamin/thesis/visualizations/unitree_nav_geom_teacher_stateful_fast_20260712/rollout_${ROLLOUT_ID}"

/home/benjamin/thesis/scripts/generated/run_unitree_geom_scan_teacher_plot.sh \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --layout-generation-attempts 10 \
  --steps 600 \
  --episode-length-s 30.0 \
  --start-clearance-resample-attempts 1 \
  --blocked-corridor-resample-attempts 1 \
  --teacher-max-vx 0.95 \
  --teacher-goal-stop-dist 0.40 \
  --teacher-geom-clearance 0.60 \
  --teacher-geom-side-penalty 8.0 \
  --teacher-geom-disengage-clear-steps 20
