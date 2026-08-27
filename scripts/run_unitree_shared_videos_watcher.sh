#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
mkdir -p artifacts/unitree_multimodal/shared_matrix_results/videos
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  scripts/render_unitree_shared_videos_sequence.py \
  >> artifacts/unitree_multimodal/shared_matrix_results/videos/watcher.log 2>&1
