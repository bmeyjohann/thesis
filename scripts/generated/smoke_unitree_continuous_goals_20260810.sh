#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=/tmp/mplconfig
export WARP_CACHE_PATH=/tmp/warp-cache
export XDG_CACHE_HOME=/tmp/unitree-cache
export MUJOCO_GL=egl

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/smoke_unitree_continuous_goals.py
