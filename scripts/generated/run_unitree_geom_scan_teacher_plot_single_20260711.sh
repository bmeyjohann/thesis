#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

ROLLOUT_ID="${1:?rollout id required}"
OUTPUT_DIR="/home/benjamin/thesis/visualizations/unitree_nav_geom_scan_teacher_10eps_20260711/rollout_${ROLLOUT_ID}"

/home/benjamin/thesis/scripts/generated/run_unitree_geom_scan_teacher_plot.sh \
  --output-dir "${OUTPUT_DIR}" \
  --layout-generation-attempts 10 \
  --steps 600 \
  --episode-length-s 30.0 \
  --start-clearance-resample-attempts 1 \
  --blocked-corridor-resample-attempts 1 \
  --teacher-goal-stop-dist 0.40 \
  --teacher-geom-clearance 0.60
