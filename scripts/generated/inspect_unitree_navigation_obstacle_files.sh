#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
for f in \
  external/unitree_rl_mjlab/src/tasks/navigation/mdp/obstacles.py \
  external/unitree_rl_mjlab/src/tasks/navigation/config/g1/env_cfgs.py \
  external/unitree_rl_mjlab/src/tasks/navigation/config/g1/safe_env_cfgs.py \
  external/unitree_rl_mjlab/src/tasks/navigation/navigation_env_cfg.py \
  external/unitree_rl_mjlab/src/tasks/navigation/mdp/observations.py \
  external/unitree_rl_mjlab/src/tasks/navigation/mdp/costs.py
do
  echo "===== $f ====="
  sed -n '1,260p' "$f"
done
