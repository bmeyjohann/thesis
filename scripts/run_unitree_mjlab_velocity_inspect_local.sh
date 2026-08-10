#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-/home/benjamin/thesis}"
UNITREE_DIR="$ROOT_DIR/external/unitree_rl_mjlab"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

TASK_ID="${TASK_ID:-Unitree-G1-Rough}"
MODEL_PATH="${MODEL_PATH:-$UNITREE_DIR/logs/rsl_rl/g1_velocity/supervisor_rough_model9999/model_9999.pt}"
MODE="${MODE:-interactive}"
VIEWER="${VIEWER:-native}"
DEVICE="${DEVICE:-cuda:0}"
NCONMAX="${NCONMAX:-256}"
HEAD_CAMERA="${HEAD_CAMERA:-0}"
CAMERA_PITCH_DOWN_DEG="${CAMERA_PITCH_DOWN_DEG:-25}"
CAMERA_FOVY="${CAMERA_FOVY:-70}"
EGOCENTRIC_VIDEO="${EGOCENTRIC_VIDEO:-}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
STEPS="${STEPS:-300}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/videos/unitree_velocity/${TASK_ID}_seed${SEED}}"
CHECKPOINT_OBSERVATION_MODE="${CHECKPOINT_OBSERVATION_MODE:-auto}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing Unitree velocity checkpoint: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/eval_unitree_velocity_policy.py" ]]; then
  echo "Missing Unitree velocity evaluator: $ROOT_DIR/eval_unitree_velocity_policy.py" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unitree-cache}"
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

echo "Starting Unitree velocity policy inspection"
echo "task:       $TASK_ID"
echo "checkpoint: $MODEL_PATH"
echo "mode:       $MODE"
echo "device:     $DEVICE"
echo "num envs:   $NUM_ENVS"
echo "seed:       $SEED"
echo "nconmax:    $NCONMAX"
echo "head cam:   $HEAD_CAMERA (pitch=${CAMERA_PITCH_DOWN_DEG}deg fovy=${CAMERA_FOVY}deg)"

CAMERA_ARGS=()
if [[ "$HEAD_CAMERA" == "1" ]]; then
  CAMERA_ARGS+=(
    --head-camera
    --camera-pitch-down-deg "$CAMERA_PITCH_DOWN_DEG"
    --camera-fovy "$CAMERA_FOVY"
  )
fi
if [[ -n "$EGOCENTRIC_VIDEO" ]]; then
  CAMERA_ARGS+=(--egocentric-video "$EGOCENTRIC_VIDEO")
fi

case "$MODE" in
  interactive)
    exec "$PYTHON_BIN" -u "$ROOT_DIR/eval_unitree_velocity_policy.py" \
      --task "$TASK_ID" \
      --checkpoint-file "$MODEL_PATH" \
      --checkpoint-observation-mode "$CHECKPOINT_OBSERVATION_MODE" \
      --viewer "$VIEWER" \
      --device "$DEVICE" \
      --num-envs "$NUM_ENVS" \
      --nconmax "$NCONMAX" \
      "${CAMERA_ARGS[@]}"
    ;;
  smoke)
    exec "$PYTHON_BIN" -u "$ROOT_DIR/eval_unitree_velocity_policy.py" \
      --task "$TASK_ID" \
      --checkpoint-file "$MODEL_PATH" \
      --device "$DEVICE" \
      --num-envs "$NUM_ENVS" \
      --steps "$STEPS" \
      --seed "$SEED" \
      --nconmax "$NCONMAX" \
      --checkpoint-observation-mode "$CHECKPOINT_OBSERVATION_MODE" \
      "${CAMERA_ARGS[@]}"
    ;;
  video)
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    exec "$PYTHON_BIN" -u "$ROOT_DIR/eval_unitree_velocity_policy.py" \
      --task "$TASK_ID" \
      --checkpoint-file "$MODEL_PATH" \
      --device "$DEVICE" \
      --num-envs 1 \
      --steps "$STEPS" \
      --seed "$SEED" \
      --nconmax "$NCONMAX" \
      --checkpoint-observation-mode "$CHECKPOINT_OBSERVATION_MODE" \
      --video \
      --video-length "$VIDEO_LENGTH" \
      --video-dir "$VIDEO_DIR" \
      "${CAMERA_ARGS[@]}"
    ;;
  *)
    echo "Unsupported MODE=$MODE; expected interactive, smoke, or video" >&2
    exit 2
    ;;
esac
