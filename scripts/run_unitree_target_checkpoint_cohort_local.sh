#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
cd "$ROOT"
exec "$PYTHON" "$ROOT/run_unitree_target_checkpoint_cohort.py" "$@"
