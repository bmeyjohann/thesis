#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
RUN_NAME=unitree_goalrl_n5_r040_linear_4k_20260715 \
SUCCESS_DIST=0.40 \
exec "$ROOT/scripts/generated/run_unitree_goal_rl_n5_r025_linear_20260715.sh"
