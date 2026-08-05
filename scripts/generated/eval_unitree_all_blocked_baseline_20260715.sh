#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: $0 direct_goal|geom_teacher|fastsac_goal_only}"
EPISODES="${2:-40}"
TEACHER_CLEARANCE="${3:-0.60}"
NUM_OBSTACLES="${4:-6}"
ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/all_blocked_baselines_20260715"
export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

common=(
  --device cuda:0 --seed 1501 --num-envs 8 --num-episodes "$EPISODES"
  --episode-length-s 60 --success-dist 0.40 --height-scan-resolution 0.25
  --goal-distance-min 2.8 --goal-distance-max 5.0
  --min-start-obstacle-clearance 1.0 --min-goal-obstacle-clearance 0.9
  --debug-goal-through-obstacle --goal-through-obstacle-prob 1.0
  --require-blocked-corridor --blocked-corridor-radius 0.45
  --blocked-corridor-ignore-end-radius 0.75 --blocked-corridor-min-cells 4
  --blocked-corridor-resample-attempts 100
  --blocked-goal-max-distance 6.5
  --debug-goal-obstacle-min-dist 0.8 --debug-goal-obstacle-max-dist 3.0
  --debug-num-obstacles "$NUM_OBSTACLES" --debug-obstacle-width-min 1.0 --debug-obstacle-width-max 1.4
  --debug-obstacle-height-min 1.0 --debug-obstacle-height-max 1.0
  --debug-terrain-rows 5 --debug-terrain-cols 10 --debug-platform-width 2.0
  --strict-min-size-obstacles --resample-terrain-tiles --output-dir "$OUT"
)

case "$METHOD" in
  direct_goal)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller direct_goal --run-name "direct_goal_n${NUM_OBSTACLES}_${EPISODES}ep" "${common[@]}"
    ;;
  geom_teacher)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller geom_scan_teacher --run-name "geom_teacher_n${NUM_OBSTACLES}_c${TEACHER_CLEARANCE}_${EPISODES}ep" \
      --teacher-geom-clearance "$TEACHER_CLEARANCE" --teacher-goal-stop-dist 0.30 "${common[@]}"
    ;;
  fastsac_goal_only)
    exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
      --controller policy --run-name "fastsac_goal_only_n${NUM_OBSTACLES}_${EPISODES}ep" \
      --model-path "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt" \
      --no-checkpoint-env-config --scan-history 1 --action-history 0 \
      --mask-height-scan --mask-goal-heading --use-layer-norm "${common[@]}"
    ;;
  *)
    echo "unsupported method: $METHOD" >&2
    exit 2
    ;;
esac
