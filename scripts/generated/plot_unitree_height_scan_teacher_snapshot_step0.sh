#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="unitree_height_scan_snapshot_step0_$(date +%Y%m%d_%H%M%S)"
export SNAPSHOT_STEPS=0
export REQUIRE_BLOCKED_CORRIDOR=0
export MIN_GOAL_OBSTACLE_CLEARANCE=0
export MIN_START_OBSTACLE_CLEARANCE=0

bash /home/benjamin/thesis/scripts/generated/plot_unitree_height_scan_teacher_snapshot.sh
