#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
printf 'pwd=%s\n' "$PWD"
find . -name 'eval_unitree_nav_baselines.py' -o -name 'train_unitree_nav_thesis.py' -o -name 'plot_unitree_nav_rollout.py' | sort
