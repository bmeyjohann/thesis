#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
CHECKPOINT="$ROOT/models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry03/step_15000.pt"
OUTPUT="$ROOT/videos/unitree_human_pref_bc_rl_best_step15000_20260807"

test -f "$CHECKPOINT"
mkdir -p "$OUTPUT"

for seed in 0 7 19; do
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
    CONTROLLER=policy \
    MODEL_PATH="$CHECKPOINT" \
    DEVICE=cuda:0 \
    NUM_ENVS=1 \
    NUM_EPISODES=1 \
    SEED="$seed" \
    RUN_NAME="pref_bc_rl_step15000_seed${seed}" \
    OUTPUT_DIR="$OUTPUT/metrics" \
    VIDEO_DIR="$OUTPUT/seed_${seed}" \
    VIDEO_LENGTH=320 \
    VIDEO_FPS=20 \
    RECORD_VIDEO=1 \
    SKIP_METRICS=1 \
    CHECKPOINT_ENV_CONFIG=1
done

echo "Rendered videos under $OUTPUT"
