#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/benjamin/thesis/visualizations/unitree_terrain_benchmarks}"
TERRAIN="${TERRAIN:-all}"
exec "$PYTHON" tools/preview_unitree_terrain_benchmarks.py \
  --terrain "$TERRAIN" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
