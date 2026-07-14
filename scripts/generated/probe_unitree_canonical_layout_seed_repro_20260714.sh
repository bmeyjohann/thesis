#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
OUT="$ROOT/logs/unitree_mjlab/layout_seed_repro_20260714"

for suffix in a b; do
  CONTROLLER=direct_goal \
  RUN_NAME="seed101_${suffix}" \
  OUTPUT_DIR="$OUT" \
  RECORD_VIDEO=0 \
  SEED=101 \
  NUM_ENVS=1 \
  NUM_EPISODES=1 \
  EPISODE_LENGTH_S=3 \
  GOAL_THROUGH_OBSTACLE_PROB=0.7 \
  GOAL_DISTANCE_MIN=2.8 \
  GOAL_DISTANCE_MAX=4.0 \
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9 \
  DEBUG_OBSTACLE_WIDTH_MIN=1.0 \
  DEBUG_OBSTACLE_WIDTH_MAX=1.4 \
  DEBUG_OBSTACLE_HEIGHT_MIN=1.0 \
  DEBUG_OBSTACLE_HEIGHT_MAX=1.0 \
  DEBUG_NUM_OBSTACLES=6 \
  DEBUG_PLATFORM_WIDTH=2.0 \
  DEBUG_TERRAIN_ROWS=5 \
  DEBUG_TERRAIN_COLS=10 \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"
done

/home/benjamin/miniconda3/envs/fasttd3/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/home/benjamin/thesis/logs/unitree_mjlab/layout_seed_repro_20260714')
episodes = [json.loads((root / f'seed101_{suffix}' / 'direct_goal_metrics.json').read_text())['episodes'][0] for suffix in ('a', 'b')]
keys = (
    'blocked_corridor',
    'blocked_corridor_cell_count',
    'nearest_corridor_obstacle_dist',
    'straight_path_length',
    'start_obstacle_clearance',
    'goal_obstacle_clearance',
)
left = {key: episodes[0][key] for key in keys}
right = {key: episodes[1][key] for key in keys}
print(json.dumps({'first': left, 'second': right, 'identical': left == right}, sort_keys=True))
if left != right:
    raise SystemExit('terrain/reset seed reproducibility probe failed')
PY
