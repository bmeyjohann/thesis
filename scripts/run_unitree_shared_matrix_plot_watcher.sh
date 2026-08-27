#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
RESULTS=artifacts/unitree_multimodal/shared_matrix_results/shared_transfer_matrix_results.csv
STATE=artifacts/unitree_multimodal/shared_matrix_results/evaluation_state.json
while [[ ! -f "$STATE" ]] || ! grep -q '"status": "completed"' "$STATE"; do sleep 60; done
/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  tools/plot_unitree_shared_matrix.py \
  --results "$RESULTS" \
  --output-dir artifacts/unitree_multimodal/shared_matrix_results/plots \
  >> artifacts/unitree_multimodal/shared_matrix_results/plot_watcher.log 2>&1

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  tools/plot_unitree_shared_summary.py \
  --students "$RESULTS" \
  --expert artifacts/unitree_multimodal/shared_matrix_results/expert_reference_results.csv \
  --output-dir artifacts/unitree_multimodal/shared_matrix_results/plots \
  >> artifacts/unitree_multimodal/shared_matrix_results/plot_watcher.log 2>&1

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  tools/plot_unitree_shared_training.py \
  --training-root artifacts/unitree_multimodal/students_shared \
  --output-dir artifacts/unitree_multimodal/shared_matrix_results/plots \
  >> artifacts/unitree_multimodal/shared_matrix_results/plot_watcher.log 2>&1
