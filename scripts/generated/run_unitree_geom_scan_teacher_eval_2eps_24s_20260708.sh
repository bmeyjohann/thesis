#!/usr/bin/env bash
set -euo pipefail
export RUN_NAME="unitree_geom_scan_teacher_astar_2eps_24s_20260708"
export NUM_EPISODES="2"
export EPISODE_LENGTH_S="24.0"
exec /home/benjamin/thesis/scripts/generated/run_unitree_geom_scan_teacher_eval.sh
