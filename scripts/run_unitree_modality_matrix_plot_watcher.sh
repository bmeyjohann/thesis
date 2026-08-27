#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
CSV=artifacts/unitree_multimodal/modality_matrix_results/modality_transfer_matrix_results.csv
while [[ ! -f "$CSV" ]]; do sleep 60; done
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  tools/plot_unitree_modality_matrix.py \
  --input "$CSV" \
  --output-dir artifacts/unitree_multimodal/modality_matrix_results
