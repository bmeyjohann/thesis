#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
GAMEPAD_ENDPOINT_FILE="${GAMEPAD_ENDPOINT_FILE:-/mnt/c/Data/thesis/local/unitree_gamepad_endpoint.json}"

if [[ -z "${LOW_LEVEL_POLICY_PATH:-}" ]]; then
  LOW_LEVEL_POLICY_PATH="$({
    find "$ROOT_DIR/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity" \
      -path '*gentle*' -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -z "$LOW_LEVEL_POLICY_PATH" || ! -f "$LOW_LEVEL_POLICY_PATH" ]]; then
  echo "Missing gentle-terrain low-level checkpoint: ${LOW_LEVEL_POLICY_PATH:-<none>}" >&2
  exit 1
fi
if [[ -z "${GAMEPAD_HOST:-}" && -f "$GAMEPAD_ENDPOINT_FILE" ]]; then
  mapfile -t endpoint_values < <(
    "$PYTHON_BIN" -c \
      'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8-sig")); print(p["host"]); print(p.get("port", 8794))' \
      "$GAMEPAD_ENDPOINT_FILE"
  )
  GAMEPAD_HOST="${endpoint_values[0]:-}"
  GAMEPAD_PORT="${GAMEPAD_PORT:-${endpoint_values[1]:-8794}}"
fi
if [[ -z "${GAMEPAD_HOST:-}" ]]; then
  echo "No Windows gamepad endpoint found." >&2
  echo "Start C:\\Data\\thesis\\scripts\\run_unitree_nav_gamepad_sender_windows.ps1 first." >&2
  exit 1
fi
GAMEPAD_PORT="${GAMEPAD_PORT:-8794}"

UNITREE_CACHE_ROOT="${UNITREE_CACHE_ROOT:-$HOME/.cache/unitree-nav}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$UNITREE_CACHE_ROOT/matplotlib}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$UNITREE_CACHE_ROOT/warp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$UNITREE_CACHE_ROOT/xdg}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

echo "Starting target-terrain manual gamepad evaluation"
echo "locomotion: $LOW_LEVEL_POLICY_PATH"
echo "gamepad:    $GAMEPAD_HOST:$GAMEPAD_PORT"
echo "terrain:    ${TARGET_TERRAIN_PRESET:-balanced}; sand drag ${SAND_LINEAR_DRAG:-120} N/(m/s)"
echo "episode:    continuous goals; ${CONTINUOUS_HORIZON_S:-3600}s environment horizon"
echo "training:   disabled"
echo "recording:  disabled"
echo "viewer:     ${VIEWER_BACKEND:-viser}"
echo "viser:      camera=${VISER_CAMERA_MODE:-chase}, scene every ${VISER_RENDER_STRIDE:-1} frame(s)"
echo "camera:     ${VIDEO_WIDTH:-192}x${VIDEO_HEIGHT:-144}, chase view, RGB every ${RGB_EVERY:-4} steps"

exec "$PYTHON_BIN" "$ROOT_DIR/eval_interactive_unitree_nav.py" \
  --controller direct_goal \
  --viewer-backend "${VIEWER_BACKEND:-viser}" \
  --viser-host "${VISER_HOST:-0.0.0.0}" \
  --viser-port "${VISER_PORT:-8080}" \
  --viser-fps "${VISER_FPS:-30}" \
  --viser-camera-mode "${VISER_CAMERA_MODE:-chase}" \
  --viser-chase-distance "${VISER_CHASE_DISTANCE:-4.0}" \
  --viser-chase-height "${VISER_CHASE_HEIGHT:-3.0}" \
  --viser-camera-lookahead "${VISER_CAMERA_LOOKAHEAD:-0.0}" \
  --viser-camera-look-height "${VISER_CAMERA_LOOK_HEIGHT:-0.85}" \
  --viser-bird-height "${VISER_BIRD_HEIGHT:-10.0}" \
  --viser-render-stride "${VISER_RENDER_STRIDE:-1}" \
  --viser-max-steps "${VISER_MAX_STEPS:-0}" \
  --viser-profile-console \
  --low-level-policy-path "$LOW_LEVEL_POLICY_PATH" \
  --target-terrain \
  --target-terrain-preset "${TARGET_TERRAIN_PRESET:-balanced}" \
  --target-terrain-seed "${TARGET_TERRAIN_SEED:-7}" \
  --target-terrain-arena-size "${TARGET_TERRAIN_ARENA_SIZE:-24}" \
  --target-terrain-sand-linear-drag "${SAND_LINEAR_DRAG:-120}" \
  --target-terrain-goal-clearance "${GOAL_CLEARANCE:-1.0}" \
  --target-terrain-goal-feature-fraction "${GOAL_FEATURE_FRACTION:-0.8}" \
  --goal-distance-min "${GOAL_DISTANCE_MIN:-4.0}" \
  --goal-distance-max "${GOAL_DISTANCE_MAX:-10.0}" \
  --success-dist "${SUCCESS_DIST:-0.4}" \
  --navigation-episode-mode continuous_goals \
  --continuous-environment-horizon-s "${CONTINUOUS_HORIZON_S:-3600}" \
  --episode-length-s "${EPISODE_LENGTH_S:-180}" \
  --height-scan-pattern grid \
  --height-scan-resolution 0.1 \
  --height-scan-forward-size 1.6 \
  --height-scan-lateral-size 1.0 \
  --device "${DEVICE:-cuda:0}" \
  --human-input-device gamepad \
  --gamepad-mode connect \
  --gamepad-host "$GAMEPAD_HOST" \
  --gamepad-port "$GAMEPAD_PORT" \
  --gamepad-intervention-mode stick \
  --gamepad-intervention-threshold "${GAMEPAD_INTERVENTION_THRESHOLD:-0.05}" \
  --gamepad-invert-lateral \
  --gamepad-debug-console \
  --no-autopilot \
  --no-start-paused \
  --auto-reset \
  --show-rgb \
  --no-rgb-debug-visualizers \
  --no-rgb-smooth-scale \
  --rgb-profile-console \
  --rgb-profile-interval 10 \
  --rgb-every "${RGB_EVERY:-4}" \
  --video-width "${VIDEO_WIDTH:-192}" \
  --video-height "${VIDEO_HEIGHT:-144}" \
  --video-camera-distance "${VIDEO_CAMERA_DISTANCE:-4.5}" \
  --video-camera-elevation "${VIDEO_CAMERA_ELEVATION:--22}" \
  --video-camera-follow-heading \
  --video-camera-heading-offset "${VIDEO_CAMERA_HEADING_OFFSET:-0}" \
  --video-max-extra-envs 0 \
  --no-video-enable-shadows \
  --no-video-enable-reflections \
  --sim-fps "${SIM_FPS:-20}" \
  --fps "${FPS:-30}" \
  --display-every "${DISPLAY_EVERY:-1}" \
  --window-width "${WINDOW_WIDTH:-1500}" \
  --window-height "${WINDOW_HEIGHT:-900}"
