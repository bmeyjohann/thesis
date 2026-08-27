#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
NO_MEMORY=artifacts/unitree_multimodal/modality_matrix_results/modality_transfer_matrix_results.csv
RECURRENT=artifacts/unitree_multimodal/recurrent_matrix_results/recurrent_transfer_matrix_results.csv
while [[ ! -f "$NO_MEMORY" || ! -f "$RECURRENT" ]]; do sleep 60; done
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  tools/plot_unitree_memory_ablation_matrix.py \
  --no-memory "$NO_MEMORY" \
  --recurrent "$RECURRENT" \
  --output-dir artifacts/unitree_multimodal/recurrent_matrix_results
