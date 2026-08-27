#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  scripts/train_unitree_recurrent_students_sequence.py "$@"
