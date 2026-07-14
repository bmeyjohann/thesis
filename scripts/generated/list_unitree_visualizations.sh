#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
find visualizations logs/unitree_mjlab -maxdepth 5 -type f \( -name '*.png' -o -name '*.json' \) \
  | grep -E 'unitree|nav|trajectory|rollout|teacher' \
  | sort \
  | tail -120
