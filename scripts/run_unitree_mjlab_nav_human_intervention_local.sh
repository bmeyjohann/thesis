#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL_PATH="${MODEL_PATH-$ROOT_DIR/models/unitree_mjlab_nav_thesis/unitree_goalonly_obstacles_forward_5k_20260729/final.pt}"
GAMEPAD_ENDPOINT_FILE="${GAMEPAD_ENDPOINT_FILE:-/mnt/c/Data/thesis/local/unitree_gamepad_endpoint.json}"

if [[ -n "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
  echo "Missing Unitree navigation checkpoint: $MODEL_PATH" >&2
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
  echo "No Windows gamepad endpoint found. Start C:\\Data\\thesis\\scripts\\run_unitree_nav_gamepad_sender_windows.ps1 first." >&2
  exit 1
fi
GAMEPAD_PORT="${GAMEPAD_PORT:-8794}"

RUN_ID="${RUN_ID:-unitree_human_$(date +%Y%m%d_%H%M%S)}"
DATASET_DIR="${HUMAN_DATASET_DIR:-$ROOT_DIR/local/unitree_human_interventions/$RUN_ID}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unitree-cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

echo "Starting Unitree human-intervention collection"
echo "checkpoint: $MODEL_PATH"
echo "environment: obstacle navigation forced on (checkpoint policy architecture retained)"
echo "gamepad:    $GAMEPAD_HOST:$GAMEPAD_PORT"
echo "takeover:   stick deflection >= ${GAMEPAD_INTERVENTION_THRESHOLD:-0.05} command norm"
echo "dataset:    $DATASET_DIR"
echo "training:   ${ONLINE_TRAIN:-0} (run=${ONLINE_RUN_NAME:-$RUN_ID})"
echo "objective:  ${ONLINE_OBJECTIVE_MODE:-pref_bc_rl}"
echo "scanner:    ${HEIGHT_SCAN_FORWARD_SIZE:-checkpoint/default}m forward x ${HEIGHT_SCAN_LATERAL_SIZE:-checkpoint/default}m lateral @ ${HEIGHT_SCAN_RESOLUTION:-checkpoint/default}m"
echo "start area: +/-${START_POSITION_RANGE:-0.8}m around each terrain-tile origin"
echo "episode mode: ${NAVIGATION_EPISODE_MODE:-continuous_goals} (goal clearance >= ${MIN_GOAL_OBSTACLE_CLEARANCE:-1.0}m)"
echo "reward:     dense distance progress x${ONLINE_DENSE_PROGRESS_SCALE:-1.0} + success ${ONLINE_SUCCESS_BONUS:-20.0} / failure ${ONLINE_FAILURE_PENALTY:--20.0}"
echo "controls:   move stick past threshold to override; left stick forward/lateral (inverted); right stick yaw"

MODEL_ARGS=()
if [[ -n "$MODEL_PATH" ]]; then
  MODEL_ARGS=(--model-path "$MODEL_PATH")
fi
ONLINE_ARGS=()
if [[ "${ONLINE_TRAIN:-0}" == "1" ]]; then
  ONLINE_ARGS=(
    --online-train
    --online-run-name "${ONLINE_RUN_NAME:-$RUN_ID}"
    --online-objective-mode "${ONLINE_OBJECTIVE_MODE:-pref_bc_rl}"
    --online-output-dir "${ONLINE_OUTPUT_DIR:-$ROOT_DIR/models/unitree_mjlab_nav_human}"
    --online-learning-starts "${ONLINE_LEARNING_STARTS:-500}"
    --online-updates-per-step "${ONLINE_UPDATES_PER_STEP:-1}"
    --online-n-step "${ONLINE_N_STEP:-5}"
    --online-actor-bc-weight "${ONLINE_ACTOR_BC_WEIGHT:-0.2}"
    --online-pref-rank-weight "${ONLINE_PREF_RANK_WEIGHT:-1.0}"
    --online-pref-proxy-value-bound "${ONLINE_PREF_PROXY_VALUE_BOUND:-1.0}"
    --online-checkpoint-interval "${ONLINE_CHECKPOINT_INTERVAL:-1000}"
    --online-dense-progress-scale "${ONLINE_DENSE_PROGRESS_SCALE:-1.0}"
    --online-success-bonus "${ONLINE_SUCCESS_BONUS:-20.0}"
    --online-failure-penalty "${ONLINE_FAILURE_PENALTY:--20.0}"
    --online-wandb-mode "${WANDB_MODE:-online}"
    --online-wandb-project "${WANDB_PROJECT:-thesis-unitree-nav-human}"
    --online-wandb-group "${WANDB_GROUP:-}"
  )
  if [[ "${ONLINE_RESTORE_FULL_STATE:-0}" == "1" ]]; then
    ONLINE_ARGS+=(--online-restore-full-state)
  else
    ONLINE_ARGS+=(--no-online-restore-full-state)
  fi
fi
STUDENT_VIEW_ARGS=()
if [[ "${STUDENT_VIEW:-1}" == "1" ]]; then
  STUDENT_VIEW_ARGS=(--student-view)
fi

exec "$PYTHON_BIN" "$ROOT_DIR/eval_interactive_unitree_nav.py" \
  --controller policy \
  "${MODEL_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --human-input-device gamepad \
  --gamepad-mode connect \
  --gamepad-host "$GAMEPAD_HOST" \
  --gamepad-port "$GAMEPAD_PORT" \
  --gamepad-intervention-mode stick \
  --gamepad-intervention-threshold "${GAMEPAD_INTERVENTION_THRESHOLD:-0.05}" \
  --gamepad-invert-lateral \
  --human-dataset-dir "$DATASET_DIR" \
  --human-dataset-chunk-size "${HUMAN_DATASET_CHUNK_SIZE:-1024}" \
  --gamepad-debug-console \
  --gamepad-status-interval-s "${GAMEPAD_STATUS_INTERVAL_S:-1.0}" \
  --fps "${FPS:-30}" \
  --sim-fps "${SIM_FPS:-20}" \
  --show-rgb \
  --show-scan-samples \
  --no-start-paused \
  --auto-reset \
  --navigation-episode-mode "${NAVIGATION_EPISODE_MODE:-continuous_goals}" \
  --continuous-environment-horizon-s "${CONTINUOUS_ENVIRONMENT_HORIZON_S:-3600}" \
  --continuous-goal-distance-min "${CONTINUOUS_GOAL_DISTANCE_MIN:-4.5}" \
  --continuous-goal-distance-max "${CONTINUOUS_GOAL_DISTANCE_MAX:-7.0}" \
  --continuous-goal-resample-attempts "${CONTINUOUS_GOAL_RESAMPLE_ATTEMPTS:-256}" \
  --continuous-goal-boundary-margin "${CONTINUOUS_GOAL_BOUNDARY_MARGIN:-0.5}" \
  --min-goal-obstacle-clearance "${MIN_GOAL_OBSTACLE_CLEARANCE:-1.0}" \
  --checkpoint-env-config \
  --height-scan-resolution "${HEIGHT_SCAN_RESOLUTION:-0.25}" \
  --height-scan-forward-size "${HEIGHT_SCAN_FORWARD_SIZE:-0.0}" \
  --height-scan-lateral-size "${HEIGHT_SCAN_LATERAL_SIZE:-0.0}" \
  --force-obstacles \
  --force-obstacle-profile strict_blocked_v1 \
  --start-position-range "${START_POSITION_RANGE:-0.8}" \
  "${STUDENT_VIEW_ARGS[@]}" \
  "${ONLINE_ARGS[@]}"
