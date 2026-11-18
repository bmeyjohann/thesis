#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
OUT=/home/benjamin/thesis/logs/unitree_goalonly_sac_rect_hist3_action20_3seed_eval_20260813
MANIFEST="$OUT/layout_manifest_100_seed81330.json"
mkdir -p "$OUT"
SEED1=/home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist3_action20_seed1_20260813_115847/final.pt
SEED2=/home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist3_action20_seed2_20260813_123350/final.pt
SEED3=/home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist3_action20_seed3_20260813_123350/final.pt
echo "[$(date -Is)] generating paired 100-episode manifest"
nice -n 15 ionice -c3 ./scripts/run_unitree_mjlab_nav_baseline_eval_local.sh CONTROLLER=policy MODEL_PATH="$SEED1" CHECKPOINT_ENV_CONFIG=1 DEVICE=cuda:0 SEED=81330 NUM_ENVS=8 NUM_EPISODES=100 EPISODE_LENGTH_S=90 OUTPUT_DIR="$OUT" RUN_NAME=manifest_generation RECORD_VIDEO=0 SKIP_METRICS=1 GENERATE_LAYOUT_MANIFEST="$MANIFEST"
for item in "seed1:$SEED1" "seed2:$SEED2" "seed3:$SEED3"; do
  name="${item%%:*}"
  model="${item#*:}"
  echo "[$(date -Is)] evaluating $name"
  nice -n 15 ionice -c3 ./scripts/run_unitree_mjlab_nav_baseline_eval_local.sh CONTROLLER=policy MODEL_PATH="$model" CHECKPOINT_ENV_CONFIG=1 DEVICE=cuda:0 SEED=81330 NUM_ENVS=8 NUM_EPISODES=100 EPISODE_LENGTH_S=90 OUTPUT_DIR="$OUT" RUN_NAME="$name" RECORD_VIDEO=0 LAYOUT_MANIFEST="$MANIFEST"
done
echo "[$(date -Is)] all evaluations finished"
