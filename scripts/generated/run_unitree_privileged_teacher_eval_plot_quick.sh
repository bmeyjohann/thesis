#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="unitree_privileged_teacher_quick_$(date +%Y%m%d_%H%M%S)"
export NUM_EPISODES=3
export REQUIRE_BLOCKED_CORRIDOR=0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.0
export MIN_START_OBSTACLE_CLEARANCE=0.0

bash /home/benjamin/thesis/scripts/generated/run_unitree_privileged_teacher_eval_plot.sh
