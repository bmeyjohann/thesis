#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  "$ROOT/tools/benchmark_unitree_layout_reset.py" \
  --device cuda:0 \
  --seed 31 \
  --episode-length-s 60 \
  --num-resets 12 \
  --persistent-rows 5 \
  --persistent-cols 10 \
  --low-level-policy-path "$ROOT/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712" \
  --debug-obstacle-width-min 1.0 \
  --debug-obstacle-width-max 1.4 \
  --debug-obstacle-height-min 1.0 \
  --debug-obstacle-height-max 1.0 \
  --debug-num-obstacles 6 \
  --debug-platform-width 2.0 \
  --output "$ROOT/logs/unitree_mjlab/persistent_layout_benchmark_20260714.json"
