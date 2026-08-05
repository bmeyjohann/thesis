#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/teacher_student_n5_final_20260715/baselines_exact"
export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

common=(
  --device cuda:0 --seed 1201 --num-envs 8 --num-episodes 40
  --episode-length-s 60 --success-dist 0.40 --height-scan-resolution 0.25
  --goal-distance-min 2.8 --goal-distance-max 4.0
  --min-start-obstacle-clearance 1.0 --min-goal-obstacle-clearance 0.9
  --debug-num-obstacles 6 --debug-obstacle-width-min 1.0 --debug-obstacle-width-max 1.4
  --debug-obstacle-height-min 1.0 --debug-obstacle-height-max 1.0
  --debug-terrain-rows 5 --debug-terrain-cols 10 --debug-platform-width 2.0
  --strict-min-size-obstacles --resample-terrain-tiles
  --output-dir "$OUT"
)

"$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller direct_goal --run-name direct_goal "${common[@]}"

"$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller geom_scan_teacher --run-name geom_teacher \
  --teacher-geom-clearance 0.60 --teacher-goal-stop-dist 0.30 "${common[@]}"

"$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller policy --run-name fastsac_goal_only \
  --model-path "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt" \
  --no-checkpoint-env-config --scan-history 1 --action-history 0 \
  --mask-height-scan --mask-goal-heading --use-layer-norm "${common[@]}"
