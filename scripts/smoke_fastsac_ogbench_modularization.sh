#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m py_compile \
  train_fast_sac_ogbench_maze.py \
  train_fast_sac_ogbench_manip.py \
  ogbench_utils/fastsac_ogbench_maze_cli.py \
  ogbench_utils/fastsac_ogbench_manip_cli.py \
  ogbench_utils/fastsac_ogbench_maze_env.py \
  ogbench_utils/fastsac_ogbench_manip_env.py \
  ogbench_utils/fastsac_ogbench_maze_train.py \
  ogbench_utils/fastsac_ogbench_manip_train.py \
  ogbench_utils/fastsac_ogbench_types.py \
  ogbench_utils/fastsac_ogbench_env.py \
  ogbench_utils/fastsac_ogbench_setup.py \
  ogbench_utils/fastsac_ogbench_loop.py \
  ogbench_utils/env_wrappers_common.py \
  ogbench_utils/intervention_wrappers.py \
  ogbench_utils/env_wrappers_maze.py \
  ogbench_utils/env_wrappers_manip.py \
  ogbench_utils/update.py \
  ogbench_utils/buffers.py \
  ogbench_utils/__init__.py

echo "[OK] FastSAC OGBench modularization compile smoke passed."
