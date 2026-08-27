#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
mkdir -p artifacts/unitree_multimodal/students_shared
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  scripts/train_unitree_shared_students_mixed_sequence.py \
  >> artifacts/unitree_multimodal/students_shared/watcher.log 2>&1
