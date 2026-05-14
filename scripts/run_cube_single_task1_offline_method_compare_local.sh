#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/scripts/offline_compare_manip_methods.py" && -f "${PWD}/scripts/offline_compare_manip_methods.py" ]]; then
  REPO_ROOT="${PWD}"
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-offline}"
ENTITY="${ENTITY:-}"
GROUP="${GROUP:-offline_method_compare}"
METHOD="${METHOD:-own}"
TEACHER_MASK_MODE="${TEACHER_MASK_MODE:-effective}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/local/ogbench_manip_datasets/cube-single-singletask-task1-v0__human_vr_online_replay__20260409_114611.npz}"
OFFLINE_UPDATES="${OFFLINE_UPDATES:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5}"
SEED="${SEED:-0}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
TS="$(date +%Y%m%d_%H%M%S)"

BASE_NAME="cube_single_task1_offline_${METHOD}"
if [[ -n "${NAME_SUFFIX}" ]]; then
  BASE_NAME="${BASE_NAME}_${NAME_SUFFIX}"
fi
NAME="${NAME:-${BASE_NAME}_${TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/offline_method_compare/${NAME}}"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${NAME}.log"

echo "=== Starting ${NAME} ==="
echo "Method: ${METHOD}"
echo "Dataset: ${DATASET_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "W&B project/group: ${PROJECT} / ${GROUP}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/offline_compare_manip_methods.py"
  --method "${METHOD}"
  --dataset_path "${DATASET_PATH}"
  --teacher_mask_mode "${TEACHER_MASK_MODE}"
  --offline_updates "${OFFLINE_UPDATES}"
  --device auto
  --env_name cube-single-singletask-task1-v0
  --obs_mode state
  --reward_type sparse
  --cube_reward_mode dense
  --no_include_goal
  --include_relative_cube_features
  --relative_only_obs
  --disable_rotation
  --teacher_type cube_markov
  --actor_hidden_dim 256
  --critic_hidden_dim 512
  --num_critics 2
  --batch_size 256
  --num_updates 1
  --cta_ratio 2
  --policy_frequency 2
  --gamma 0.97
  --tau 0.005
  --alpha_init 0.001
  --fixed_alpha 0.001
  --alpha_min 0.001
  --alpha_max 0.001
  --use_layer_norm
  --pref_rank_weight 1.0
  --pref_rank_margin 0.01
  --pref_loss_type hinge
  --pref_critic_scope all
  --pref_linked_action_epsilon 0.01
  --pref_linked_action_weight_scale 0.25
  --demo_sample_ratio 0.5
  --eil_bad_pre_steps 8
  --eval_interval "${EVAL_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --seed "${SEED}"
  --name "${NAME}"
  --exp_name "${NAME}"
  --output_dir "${OUTPUT_DIR}"
  --use_wandb
  --project "${PROJECT}"
  --group "${GROUP}"
)
if [[ -n "${ENTITY}" ]]; then
  CMD+=(--entity "${ENTITY}")
fi
if [[ -n "${WANDB_MODE_VALUE}" ]]; then
  CMD+=(--wandb_mode "${WANDB_MODE_VALUE}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${NAME} ==="
