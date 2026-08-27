#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
mkdir -p artifacts/unitree_multimodal/shared_matrix_results
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  scripts/evaluate_unitree_shared_matrix_sequence.py \
  >> artifacts/unitree_multimodal/shared_matrix_results/watcher.log 2>&1
