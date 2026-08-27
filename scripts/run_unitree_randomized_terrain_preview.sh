#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/benjamin/thesis/visualizations/unitree_randomized_terrain_suite}"
exec "$PYTHON" tools/preview_unitree_randomized_terrain_suite.py \
  --output-dir "$OUTPUT_DIR" \
  "$@"
