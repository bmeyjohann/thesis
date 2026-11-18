#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis/artifacts/unitree_bc_blocked_20261012_r4/repo
export UNITREE_LOCAL_SMOKE=1
export UNITREE_STUDY_ROOT=/home/benjamin/thesis/artifacts/unitree_bc_blocked_20261012_r4
export PYTHONUNBUFFERED=1
exec /usr/bin/timeout --signal=TERM --kill-after=60s 258000 \
  /home/benjamin/miniconda3/envs/fasttd3/bin/python -u \
  /home/benjamin/thesis/artifacts/unitree_bc_blocked_20261012_r4/repo/scripts/cluster/unitree_blocked_bc_study.py run \
  >> /home/benjamin/thesis/artifacts/unitree_bc_blocked_20261012_r4/logs/controller.log 2>&1
