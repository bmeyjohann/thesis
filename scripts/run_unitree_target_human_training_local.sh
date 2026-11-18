#!/usr/bin/env bash
set -euo pipefail
exec 9>/tmp/unitree_core_gpu0.lock
flock -n 9 || { echo 'Another Unitree GPU job owns the shared lock.' >&2; exit 1; }
ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
LOW_LEVEL=${LOW_LEVEL_POLICY_PATH:-$ROOT/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-09-03_01-25-36_g1_gentle_turn_curriculum_seed0_resume13000_20260903/model_15999.pt}
ENDPOINT=${GAMEPAD_ENDPOINT_FILE:-/mnt/c/Data/thesis/local/unitree_gamepad_endpoint.json}
SENSORS=${NAVIGATION_OBSERVATION_MODE:-height_scan}
ENCODER=${NAVIGATION_ENCODER_CHECKPOINT:-$ROOT/artifacts/unitree_multimodal_target/gentle_gru_reconstruction_shared_seed1_100k_20260903_v3/student_100000.pt}
if [[ "$SENSORS" == php_depth && -z "${NAVIGATION_ENCODER_CHECKPOINT:-}" ]]; then
  ENCODER=$ROOT/artifacts/unitree_depth_transfer_20261013/models/php_student/depth_backbone.onnx
fi
RUN_NAME=${ONLINE_RUN_NAME:-target_human_${SENSORS}_$(date +%Y%m%d_%H%M%S)}
read -r -a FEATURE_LAGS <<< "${NAVIGATION_FEATURE_LAGS:-0 1 25}"
VIEW_ARGS=()
if [[ "$SENSORS" == rgb_height || "$SENSORS" == php_depth ]]; then
  VIEW_ARGS+=(--observation-only)
fi
test -f "$LOW_LEVEL"
if [[ "$SENSORS" != height_scan ]]; then test -f "$ENCODER"; fi
if [[ -z "${GAMEPAD_HOST:-}" ]]; then
  mapfile -t connection < <("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8-sig")); print(d["host"]); print(d["port"])' "$ENDPOINT")
  GAMEPAD_HOST=${connection[0]}
  GAMEPAD_PORT=${connection[1]}
fi
export MUJOCO_GL=${MUJOCO_GL:-egl}
cd "$ROOT"
exec "$PYTHON" eval_interactive_unitree_nav.py \
  --viewer-backend viser --controller policy --online-train --autopilot \
  "${VIEW_ARGS[@]}" \
  --navigation-observation-mode "$SENSORS" --navigation-encoder-checkpoint "$ENCODER" \
  --navigation-image-profile "${NAVIGATION_IMAGE_PROFILE:-stabilized_goal_v3}" \
  --navigation-feature-lags "${FEATURE_LAGS[@]}" \
  --online-objective-mode "${ONLINE_OBJECTIVE_MODE:-pref_bc_rl}" \
  --online-lr-actor "${ONLINE_LR_ACTOR:-0.0001}" --online-lr-critic "${ONLINE_LR_CRITIC:-0.0003}" \
  --online-batch-size "${ONLINE_BATCH_SIZE:-256}" --online-actor-bc-weight "${ONLINE_ACTOR_BC_WEIGHT:-0.1}" \
  --online-learning-starts 128 --online-policy-frequency 2 --online-updates-per-step 1 \
  --online-alpha "${ONLINE_ALPHA:-0.01}" \
  --online-alpha-min "${ONLINE_ALPHA_MIN:-0.00001}" \
  --online-alpha-max "${ONLINE_ALPHA_MAX:-0.1}" \
  --online-alpha-lr "${ONLINE_ALPHA_LR:-0.0003}" \
  --online-alpha-state-scope "${ONLINE_ALPHA_STATE_SCOPE:-student}" \
  --online-target-entropy "${ONLINE_TARGET_ENTROPY:--3}" \
  --online-pref-sampling-mode "${ONLINE_PREF_SAMPLING_MODE:-separate}" \
  --online-pref-batch-size "${ONLINE_PREF_BATCH_SIZE:-256}" \
  --online-pref-replay-capacity "${ONLINE_PREF_REPLAY_CAPACITY:-100000}" \
  --online-pref-lambda-init "${ONLINE_PREF_LAMBDA_INIT:-1.0}" \
  --online-pref-lambda-lr "${ONLINE_PREF_LAMBDA_LR:-0.0}" \
  --online-pref-lambda-max "${ONLINE_PREF_LAMBDA_MAX:-10.0}" \
  --online-run-name "$RUN_NAME" --online-total-steps "${ONLINE_TOTAL_STEPS:-4500}" \
  --online-session-receipt "${ONLINE_SESSION_RECEIPT:-}" \
  --online-checkpoint-interval 500 --online-wandb-mode offline \
  --online-dense-progress-scale 1 --online-success-bonus 20 \
  --online-failure-penalty -20 --online-n-step "${ONLINE_N_STEP:-1}" \
  --human-dataset-dir "$ROOT/local/unitree_human_interventions/$RUN_NAME" \
  --low-level-policy-path "$LOW_LEVEL" --num-envs 1 --device cuda:0 --seed "${SEED:-7}" \
  --target-terrain --target-terrain-preset "${TARGET_TERRAIN_PRESET:-challenge}" --target-terrain-seed "${SEED:-7}" \
  --target-terrain-dynamic-obstacles --target-terrain-material-resolution 0.2 \
  --target-terrain-ice-friction 0.06 --target-terrain-ice-contact-priority 1 \
  --target-terrain-arena-size 24 --target-terrain-sand-linear-drag 120 \
  --target-terrain-goal-clearance 1 --target-terrain-goal-feature-fraction 0.8 \
  --goal-distance-min 4 --goal-distance-max 10 --success-dist 0.4 \
  --navigation-episode-mode continuous_goals --continuous-environment-horizon-s 3600 \
  --episode-length-s 180 --navigation-policy-hz 5 \
  --height-scan-pattern grid --height-scan-resolution 0.1 \
  --height-scan-forward-size 1.6 --height-scan-lateral-size 1.0 \
  --scan-history 1 --action-history 0 --policy-encoder mlp --hidden-dim 256 --use-layer-norm \
  --mask-goal-heading --goal-encoding distance_bearing \
  --human-input-device gamepad --gamepad-mode connect \
  --gamepad-host "$GAMEPAD_HOST" --gamepad-port "${GAMEPAD_PORT:-8794}" \
  --gamepad-intervention-mode stick --gamepad-intervention-threshold 0.05 \
  --gamepad-invert-lateral --gamepad-debug-console \
  --show-rgb --video-width 192 --video-height 144 --rgb-every 4 \
  --video-camera-follow-heading --video-camera-distance 4.5 --video-camera-elevation -22 \
  --no-rgb-debug-visualizers --no-video-enable-shadows --no-video-enable-reflections \
  --sim-fps 20 --fps 30 --start-paused --auto-reset "$@"
