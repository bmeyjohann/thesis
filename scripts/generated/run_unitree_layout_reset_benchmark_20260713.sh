#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unitree-cache}"
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/tools/benchmark_unitree_layout_reset.py \
  --device cuda:0 \
  --low-level-policy-path /home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712 \
  --output /home/benjamin/thesis/visualizations/unitree_nav_layout_benchmark_20260713/results.json
