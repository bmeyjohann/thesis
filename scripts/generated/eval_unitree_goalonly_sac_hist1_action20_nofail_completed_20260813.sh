#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
OUT=/home/benjamin/thesis/logs/unitree_goalonly_sac_hist1_action20_nofail_paired100_20260813
MANIFEST=/home/benjamin/thesis/logs/unitree_goalonly_sac_rect_hist3_action20_3seed_eval_20260813/layout_manifest_100_seed81330.json
mkdir -p "$OUT"
run_eval() {
  name="$1"; model="$2"
  echo "[$(date -Is)] evaluating $name"
  nice -n 15 ionice -c3 ./scripts/run_unitree_mjlab_nav_baseline_eval_local.sh CONTROLLER=policy MODEL_PATH="$model" CHECKPOINT_ENV_CONFIG=1 DEVICE=cuda:0 SEED=81330 NUM_ENVS=8 NUM_EPISODES=100 EPISODE_LENGTH_S=90 OUTPUT_DIR="$OUT" RUN_NAME="$name" RECORD_VIDEO=0 LAYOUT_MANIFEST="$MANIFEST"
}
run_eval sac_hist1_nofail_seed1 /home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist1_action20_nofail_seed1_20260813_163309/final.pt
run_eval sac_hist1_nofail_seed2 /home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist1_action20_nofail_seed2_20260813_163309/final.pt
echo "[$(date -Is)] completed SAC evaluations finished"
