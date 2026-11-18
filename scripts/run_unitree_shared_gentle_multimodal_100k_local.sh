#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
EXPERT="$ROOT/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-09-03_01-25-36_g1_gentle_turn_curriculum_seed0_resume13000_20260903/model_15999.pt"
OUTPUT="$ROOT/artifacts/unitree_multimodal_target/gentle_gru_reconstruction_shared_seed1_100k_20260903_v3"

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$HOME/.cache/unitree-nav/warp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache/unitree-nav/xdg}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$WARP_CACHE_PATH" "$XDG_CACHE_HOME" "$OUTPUT"

exec "$PYTHON" "$ROOT/train_unitree_shared_multimodal_student.py" \
  --architecture gru_reconstruction \
  --geometry mixed \
  --material mixed \
  --expert-checkpoint "$EXPERT" \
  --output "$OUTPUT" \
  --seed 1 \
  --num-envs 32 \
  --steps 100000 \
  --unroll-length 50 \
  --learning-rate 3e-4 \
  --reconstruction-weight 0.10 \
  --modality-sampling mixed_subsets \
  --modality-hold-steps 250 \
  --camera-width 80 \
  --camera-height 60 \
  --camera-max-depth 5.0 \
  --checkpoint-interval 2500
