#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/scripts/recover_offline_actor_manip_from_checkpoint.py" && -f "${PWD}/scripts/recover_offline_actor_manip_from_checkpoint.py" ]]; then
  REPO_ROOT="${PWD}"
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/models/fast_sac/cube_single_task1_human_collect_norot_fixedalpha1e3_20260407_185302/cube_single_singletask_task1_v0_step10000.pt}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/local/ogbench_manip_datasets/cube-single-singletask-task1-v0__human_vr_online_replay__20260407_185310.npz}"
NUM_GRADIENT_STEPS="${NUM_GRADIENT_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
ACTION_L2_WEIGHT="${ACTION_L2_WEIGHT:-1e-3}"
BC_TEACHER_WEIGHT="${BC_TEACHER_WEIGHT:-0.0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"
NAME="${NAME:-offline_actor_recover_qonly}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/offline_actor_recover/${NAME}}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${NAME}.log"

echo "=== Starting ${NAME} ==="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/recover_offline_actor_manip_from_checkpoint.py" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --device "${DEVICE}" \
  --num_gradient_steps "${NUM_GRADIENT_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --actor_lr "${ACTOR_LR}" \
  --action_l2_weight "${ACTION_L2_WEIGHT}" \
  --bc_teacher_weight "${BC_TEACHER_WEIGHT}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --eval_num_envs "${EVAL_NUM_ENVS}" \
  --seed "${SEED}" \
  --name "${NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${NAME} ==="
