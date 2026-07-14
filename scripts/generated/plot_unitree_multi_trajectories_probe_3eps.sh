#!/usr/bin/env bash
set -euo pipefail

export NUM_EPISODES=3
export RUN_NAME="unitree_multi_traj_round_3eps_$(date +%Y%m%d_%H%M%S)"
exec /home/benjamin/thesis/scripts/generated/plot_unitree_multi_trajectories_probe.sh
