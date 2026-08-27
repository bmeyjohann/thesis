#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  scripts/evaluate_unitree_recurrent_matrix_sequence.py "$@"
