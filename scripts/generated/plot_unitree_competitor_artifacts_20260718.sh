#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
SEED="${1:-0}"
TAG="${2:-20260718_v2}"
OUT="$ROOT/visualizations/unitree_competitors_20260718"
mkdir -p "$OUT"

for method in thesis hilserl eil pvp hg_dagger sac; do
  checkpoint="$ROOT/models/unitree_mjlab_nav_thesis/unitree_compare_${method}_seed${SEED}_${TAG}/final.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Skipping $method: missing $checkpoint" >&2
    continue
  fi
  if [[ -f "$OUT/trajectories/$method/policy_topdown_rollout_01.png" && \
        -f "$OUT/trajectories/$method/policy_topdown_rollout_02.png" ]]; then
    echo "Skipping $method: two trajectory rollouts already exist"
    continue
  fi
  REPO_ROOT="$ROOT" \
  CONTROLLER=policy \
  MODEL_PATH="$checkpoint" \
  CHECKPOINT_ENV_CONFIG=1 \
  NUM_ROLLOUTS=2 \
  STEPS=1800 \
  SEED=941 \
  OUTPUT_DIR="$OUT/trajectories/$method" \
    "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"
done

MPLCONFIGDIR=/tmp/mplconfig \
  /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  "$ROOT/tools/plot_unitree_competitor_comparison.py" \
  --models-root "$ROOT/models/unitree_mjlab_nav_thesis" \
  --output-dir "$OUT" \
  --seed "$SEED" \
  --run-suffix "$TAG"
