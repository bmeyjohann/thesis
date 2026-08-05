#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: $0 goal_only|student|student_smooth07|teacher_raw|teacher_smooth04|teacher_commit|teacher_stable [episodes]}"
EPISODES="${2:-24}"
MODEL_OVERRIDE="${3:-}"
RUN_LABEL="${4:-student_step1000}"
ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/random_multiblocked_20260716"
GOAL_MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt"
STUDENT_MODEL="${MODEL_OVERRIDE:-$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_allblocked_n1_max65_n5_combined_5000_20260715/step_1000.pt}"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

common=(
  --device cuda:0 --seed 1607 --num-envs 8 --num-episodes "$EPISODES"
  --episode-length-s 90 --success-dist 0.40 --height-scan-resolution 0.25
  --goal-distance-min 4.5 --goal-distance-max 8.0
  --min-start-obstacle-clearance 1.0 --min-goal-obstacle-clearance 0.9
  --debug-goal-through-obstacle --goal-through-obstacle-prob 1.0
  --require-blocked-corridor --blocked-corridor-radius 0.45
  --blocked-corridor-ignore-end-radius 0.75 --blocked-corridor-min-cells 4
  --blocked-corridor-resample-attempts 200 --blocked-goal-max-distance 8.0
  --blocked-goal-distance-sampling uniform
  --debug-goal-obstacle-min-dist 1.0 --debug-goal-obstacle-max-dist 5.5
  --debug-num-obstacles 6 --debug-obstacle-width-min 1.0 --debug-obstacle-width-max 1.4
  --debug-obstacle-height-min 1.0 --debug-obstacle-height-max 1.0
  --debug-terrain-rows 5 --debug-terrain-cols 10 --debug-platform-width 2.0
  --strict-min-size-obstacles --resample-terrain-tiles --output-dir "$OUT"
)

case "$METHOD" in
  goal_only)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller policy --model-path "$GOAL_MODEL" --no-checkpoint-env-config \
      --scan-history 1 --action-history 0 --mask-height-scan --mask-goal-heading \
      --use-layer-norm --run-name "goal_only_${EPISODES}ep" "${common[@]}"
    ;;
  student)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller policy --model-path "$STUDENT_MODEL" --no-checkpoint-env-config \
      --scan-history 1 --action-history 0 --mask-goal-heading --use-layer-norm \
      --run-name "${RUN_LABEL}_${EPISODES}ep" "${common[@]}"
    ;;
  student_smooth07)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller policy --model-path "$STUDENT_MODEL" --no-checkpoint-env-config \
      --scan-history 1 --action-history 0 --mask-goal-heading --use-layer-norm \
      --policy-action-smoothing 0.70 \
      --run-name "${RUN_LABEL}_smooth07_${EPISODES}ep" "${common[@]}"
    ;;
  teacher_raw)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller geom_scan_teacher --teacher-geom-clearance 0.50 \
      --teacher-goal-stop-dist 0.30 --run-name "teacher_raw_${EPISODES}ep" "${common[@]}"
    ;;
  teacher_stable)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller geom_scan_teacher --teacher-geom-clearance 0.50 \
      --teacher-geom-command-smoothing 0.70 \
      --teacher-geom-waypoint-commit-distance 0.50 \
      --teacher-geom-waypoint-reach-dist 0.25 \
      --teacher-goal-stop-dist 0.30 --run-name "teacher_stable_${EPISODES}ep" "${common[@]}"
    ;;
  teacher_smooth04)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller geom_scan_teacher --teacher-geom-clearance 0.50 \
      --teacher-geom-command-smoothing 0.40 \
      --teacher-goal-stop-dist 0.30 --run-name "teacher_smooth04_${EPISODES}ep" "${common[@]}"
    ;;
  teacher_commit)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller geom_scan_teacher --teacher-geom-clearance 0.50 \
      --teacher-geom-waypoint-commit-distance 0.50 \
      --teacher-geom-waypoint-reach-dist 0.25 \
      --teacher-goal-stop-dist 0.30 --run-name "teacher_commit_${EPISODES}ep" "${common[@]}"
    ;;
  *)
    echo "Unsupported method: $METHOD" >&2
    exit 2
    ;;
esac
