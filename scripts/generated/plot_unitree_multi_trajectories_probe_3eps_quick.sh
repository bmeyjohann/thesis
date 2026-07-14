#!/usr/bin/env bash
set -euo pipefail

export NUM_EPISODES=3
export RUN_NAME="unitree_multi_traj_round_3eps_quick_$(date +%Y%m%d_%H%M%S)"
export REQUIRE_BLOCKED_CORRIDOR=0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.0
export MIN_START_OBSTACLE_CLEARANCE=0.0

bash /home/benjamin/thesis/scripts/generated/plot_unitree_multi_trajectories_probe.sh
