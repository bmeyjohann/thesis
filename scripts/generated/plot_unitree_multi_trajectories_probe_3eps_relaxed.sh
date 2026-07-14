#!/usr/bin/env bash
set -euo pipefail

export NUM_EPISODES=3
export RUN_NAME="unitree_multi_traj_round_3eps_relaxed_$(date +%Y%m%d_%H%M%S)"
export MIN_GOAL_OBSTACLE_CLEARANCE=0.6
export MIN_START_OBSTACLE_CLEARANCE=0.65
export GOAL_CLEARANCE_RESAMPLE_ATTEMPTS=200
export START_CLEARANCE_RESAMPLE_ATTEMPTS=100
export BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS=200

bash /home/benjamin/thesis/scripts/generated/plot_unitree_multi_trajectories_probe.sh
