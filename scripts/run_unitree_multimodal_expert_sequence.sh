#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/run_unitree_multimodal_expert_sequence.py "$@"
