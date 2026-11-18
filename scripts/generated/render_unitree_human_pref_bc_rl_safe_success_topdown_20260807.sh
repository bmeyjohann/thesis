#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
CHECKPOINT="$ROOT/models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry03/step_15000.pt"
MANIFEST="$ROOT/local/eval_manifests/unitree_human_rectscan_blocked_100_seed0.json"
OUTPUT="$ROOT/videos/unitree_human_pref_bc_rl_best_step15000_safe_success_topdown_20260807"

test -f "$CHECKPOINT"
test -f "$MANIFEST"
mkdir -p "$OUTPUT"

# These episodes were collision-free successes in the fixed 100-episode
# benchmark. Replaying their manifest entries makes the videos reproducible.
for episode in 8 26 52 73 82; do
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
    CONTROLLER=policy \
    MODEL_PATH="$CHECKPOINT" \
    DEVICE=cuda:0 \
    NUM_ENVS=1 \
    NUM_EPISODES=1 \
    RUN_NAME="pref_bc_rl_step15000_manifest_ep${episode}" \
    OUTPUT_DIR="$OUTPUT/metrics" \
    VIDEO_DIR="$OUTPUT/episode_${episode}" \
    VIDEO_LENGTH=320 \
    VIDEO_FPS=20 \
    VIDEO_MANIFEST_INDEX="$episode" \
    VIDEO_STOP_ON_DONE=1 \
    VIDEO_CAMERA_DISTANCE=8.0 \
    VIDEO_CAMERA_ELEVATION=-70.0 \
    VIDEO_CAMERA_AZIMUTH=90.0 \
    VIDEO_WIDTH=640 \
    VIDEO_HEIGHT=640 \
    LAYOUT_MANIFEST="$MANIFEST" \
    RECORD_VIDEO=1 \
    SKIP_METRICS=1 \
    CHECKPOINT_ENV_CONFIG=1
done

echo "Rendered top-down videos under $OUTPUT"
