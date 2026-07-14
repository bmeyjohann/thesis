#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

LABEL="${1:?label required}"
POLICY_PATH="${2:?policy path required}"

MPLCONFIGDIR=/tmp/mplconfig \
MUJOCO_GL=egl \
WARP_CACHE_PATH=/tmp/warp-cache \
XDG_CACHE_HOME=/tmp/unitree-cache \
/home/benjamin/miniconda3/envs/fasttd3/bin/python -u \
  /home/benjamin/thesis/tools/eval_unitree_locomotion_velocity.py \
  --low-level-policy-path "${POLICY_PATH}" \
  --device cuda:0 \
  --num-envs 64 \
  --warmup-s 2.0 \
  --duration-s 6.0 \
  --output "/home/benjamin/thesis/logs/unitree_mjlab/locomotion_velocity_${LABEL}_20260712.json"
