#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/evaluate_unitree_modality_matrix_sequence.py "$@"
