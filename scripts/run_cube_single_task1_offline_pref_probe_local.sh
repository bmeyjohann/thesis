#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/scripts/probe_offline_pref_critic_manip.py" && -f "${PWD}/scripts/probe_offline_pref_critic_manip.py" ]]; then
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
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/models/fast_sac/cube_single_task1_human_onlycollecteddemos_demoaug/cube_single_singletask_task1_v0_final.pt}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/local/ogbench_manip_datasets/cube-single-singletask-task1-v0__human_vr_online_replay__20260409_114611.npz}"
DEVICE="${DEVICE:-auto}"
TEACHER_MASK_MODE="${TEACHER_MASK_MODE:-effective}"
ACTOR_INIT_MODE="${ACTOR_INIT_MODE:-checkpoint}"
CRITIC_STEPS="${CRITIC_STEPS:-500}"
ACTOR_STEPS="${ACTOR_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
PREF_LOSS_TYPE="${PREF_LOSS_TYPE:-hinge}"
PREF_MARGIN="${PREF_MARGIN:-0.01}"
PREF_WEIGHT="${PREF_WEIGHT:-1.0}"
LINKED_ACTION_FILTER_MODE="${LINKED_ACTION_FILTER_MODE:-epsilon}"
LINKED_ACTION_SCOPE="${LINKED_ACTION_SCOPE:-all}"
LINKED_ACTION_FILTER_METRIC="${LINKED_ACTION_FILTER_METRIC:-l1}"
LINKED_ACTION_WEIGHT_METRIC="${LINKED_ACTION_WEIGHT_METRIC:-mean_abs}"
LINKED_ACTION_EPSILON="${LINKED_ACTION_EPSILON:-1e-6}"
LINKED_ACTION_WEIGHT_SCALE="${LINKED_ACTION_WEIGHT_SCALE:-0.0}"
LINKED_ACTION_ANGLE_THRESHOLD_DEG="${LINKED_ACTION_ANGLE_THRESHOLD_DEG:--1}"
LINKED_COMPONENT_XYZ_THRESHOLD="${LINKED_COMPONENT_XYZ_THRESHOLD:--1}"
LINKED_COMPONENT_YAW_THRESHOLD="${LINKED_COMPONENT_YAW_THRESHOLD:--1}"
LINKED_COMPONENT_GRIPPER_THRESHOLD="${LINKED_COMPONENT_GRIPPER_THRESHOLD:--1}"
LINKED_COMPONENT_ADAPTIVE_ENABLE="${LINKED_COMPONENT_ADAPTIVE_ENABLE:-0}"
LINKED_COMPONENT_NEAR_DISTANCE="${LINKED_COMPONENT_NEAR_DISTANCE:--1}"
LINKED_COMPONENT_FAR_DISTANCE="${LINKED_COMPONENT_FAR_DISTANCE:--1}"
LINKED_COMPONENT_NEAR_SCALE="${LINKED_COMPONENT_NEAR_SCALE:--1}"
ACTOR_Q_WEIGHT="${ACTOR_Q_WEIGHT:-1.0}"
BC_TEACHER_WEIGHT="${BC_TEACHER_WEIGHT:-0.0}"
ACTION_L2_WEIGHT="${ACTION_L2_WEIGHT:-1e-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5}"
SEED="${SEED:-0}"
NAME="${NAME:-offline_pref_probe}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/offline_actor_recover/${NAME}}"
USE_WANDB="${USE_WANDB:-1}"
PROJECT="${PROJECT:-ogbench-manip-offline}"
ENTITY="${ENTITY:-}"
GROUP="${GROUP:-}"
WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${NAME}.log"

echo "=== Starting ${NAME} ==="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Linked action filter: mode=${LINKED_ACTION_FILTER_MODE} scope=${LINKED_ACTION_SCOPE} metric=${LINKED_ACTION_FILTER_METRIC}"
echo "Linked action epsilon / weight scale / weight metric: ${LINKED_ACTION_EPSILON} / ${LINKED_ACTION_WEIGHT_SCALE} / ${LINKED_ACTION_WEIGHT_METRIC}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/probe_offline_pref_critic_manip.py"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --dataset_path "${DATASET_PATH}"
  --device "${DEVICE}"
  --teacher_mask_mode "${TEACHER_MASK_MODE}"
  --actor_init_mode "${ACTOR_INIT_MODE}"
  --critic_steps "${CRITIC_STEPS}"
  --actor_steps "${ACTOR_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --critic_lr "${CRITIC_LR}"
  --actor_lr "${ACTOR_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --pref_loss_type "${PREF_LOSS_TYPE}"
  --pref_margin "${PREF_MARGIN}"
  --pref_weight "${PREF_WEIGHT}"
  --linked_action_filter_mode "${LINKED_ACTION_FILTER_MODE}"
  --linked_action_scope "${LINKED_ACTION_SCOPE}"
  --linked_action_filter_metric "${LINKED_ACTION_FILTER_METRIC}"
  --linked_action_weight_metric "${LINKED_ACTION_WEIGHT_METRIC}"
  --linked_action_epsilon "${LINKED_ACTION_EPSILON}"
  --linked_action_weight_scale "${LINKED_ACTION_WEIGHT_SCALE}"
  --linked_action_angle_threshold_deg "${LINKED_ACTION_ANGLE_THRESHOLD_DEG}"
  --linked_component_xyz_threshold "${LINKED_COMPONENT_XYZ_THRESHOLD}"
  --linked_component_yaw_threshold "${LINKED_COMPONENT_YAW_THRESHOLD}"
  --linked_component_gripper_threshold "${LINKED_COMPONENT_GRIPPER_THRESHOLD}"
  --linked_component_near_distance "${LINKED_COMPONENT_NEAR_DISTANCE}"
  --linked_component_far_distance "${LINKED_COMPONENT_FAR_DISTANCE}"
  --linked_component_near_scale "${LINKED_COMPONENT_NEAR_SCALE}"
  --actor_q_weight "${ACTOR_Q_WEIGHT}"
  --bc_teacher_weight "${BC_TEACHER_WEIGHT}"
  --action_l2_weight "${ACTION_L2_WEIGHT}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --seed "${SEED}"
  --name "${NAME}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ "${USE_WANDB}" == "1" ]]; then
  CMD+=(--use_wandb --project "${PROJECT}")
fi
if [[ -n "${ENTITY}" ]]; then
  CMD+=(--entity "${ENTITY}")
fi
if [[ -n "${GROUP}" ]]; then
  CMD+=(--group "${GROUP}")
fi
if [[ -n "${WANDB_MODE_OVERRIDE}" ]]; then
  CMD+=(--wandb_mode "${WANDB_MODE_OVERRIDE}")
fi
if [[ "${LINKED_COMPONENT_ADAPTIVE_ENABLE}" == "1" ]]; then
  CMD+=(--linked_component_adaptive_enable)
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${NAME} ==="
