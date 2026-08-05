#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
RUN_NAME=unitree_goalrl_n5_r040_exp4_t1_4k_20260715 \
SUCCESS_DIST=0.40 \
LEARNER_REWARD_MODE=dense_progress_exp \
DENSE_PROGRESS_EXP_SCALE=4.0 \
DENSE_PROGRESS_EXP_TEMPERATURE=1.0 \
exec "$ROOT/scripts/generated/run_unitree_goal_rl_n5_r025_linear_20260715.sh"
