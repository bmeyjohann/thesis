#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT=/home/benjamin/thesis
RUN="$ROOT/scripts/run_unitree_target_teacher_benchmark_local.sh"
OUT="$ROOT/artifacts/target_hazard_gate_20260914/audit_v2"
cd "$ROOT"
bash "$RUN" --hazard-progress --direct-goal --max-seconds 180 --output "$OUT/direct" --resume
bash "$RUN" --hazard-progress --probe-controller random --seeds 201 202 203 204 209 217 222 226 --max-seconds 180 --output "$OUT/random" --resume
bash "$RUN" --hazard-progress --probe-controller stationary --seeds 201 202 203 204 209 217 222 226 --max-seconds 180 --output "$OUT/stationary" --resume
bash "$RUN" --hazard-progress --nav-checkpoint "$ROOT/artifacts/target_teacher_screen_20260913/online_rgb_height/training/final.pt" --max-seconds 180 --output "$OUT/student" --resume
