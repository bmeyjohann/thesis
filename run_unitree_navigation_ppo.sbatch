#!/bin/bash
#SBATCH --job-name=unitree_go2_nav
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<optional_qos>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/unitree_go2_nav_%j.out
#SBATCH --error=logs/unitree_go2_nav_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SAFE_LOCO_ROOT=${SAFE_LOCO_ROOT:-${REPO_ROOT}/safe-locomotion}

echo "=========================================="
echo "Navigation training job starting"
echo "Working directory: ${REPO_ROOT}"
echo "Safe Locomotion root: ${SAFE_LOCO_ROOT}"
echo "=========================================="

cd "${REPO_ROOT}"

mkdir -p logs wandb

export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-${REPO_ROOT}/unitree_ros/unitree_ros}"
export GIT_PYTHON_REFRESH=quiet
unset CUDA_VISIBLE_DEVICES

cd "${REPO_ROOT}/sc_venv_template_isaaclab"
source activate.sh
cd "${REPO_ROOT}"

source "${REPO_ROOT}/.venv/bin/activate"

export XDG_CACHE_HOME=/p/project1/hai_1074/meyjohann1/apptainer/cache
export XDG_DATA_HOME=/p/project1/hai_1074/meyjohann1/apptainer/data
export KIT_USER_ROOT=/p/project1/hai_1074/meyjohann1/apptainer/kit
export KIT_USER_DATA=/p/project1/hai_1074/meyjohann1/apptainer/data
export KIT_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache
export KIT_DERIVED_DATA_CACHE=/p/project1/hai_1074/meyjohann1/apptainer/DerivedDataCache
export KIT_PIP_INSTALL_PATH=/p/project1/hai_1074/meyjohann1/apptainer/pip3-envs
export OMNI_USER_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache
export OMNI_KIT_USER_DIR=/p/project1/hai_1074/meyjohann1/apptainer/kit
export OMNI_KIT_DATA_DIR=/p/project1/hai_1074/meyjohann1/apptainer/data
export OMNI_KIT_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache

export OMNI_DISABLE_EXTENSIONS="omni.iray.libs,omni.mdl.neuraylib,omni.kit.usd.mdl,carb.scenerenderer-rtx.plugin,omni.rtx.*,omni.hydra.rtx"

export WANDB_MODE=offline
export WANDB_DIR="${REPO_ROOT}/wandb"
export WANDB_START_METHOD=thread

ISAAC_PYTHON=${ISAAC_PYTHON:-~/meyjohann1/isaacsim5.1/python.sh}
TRAIN_PY=${TRAIN_PY:-${SAFE_LOCO_ROOT}/scripts/rsl_rl/train.py}
TASK=${TASK:-Isaac-Navigation-Flat-Go2-v0}
ALGO=${ALGO:-ppo}
RUN_NAME=${RUN_NAME:-navigation_ppo_${SLURM_JOB_ID}}
LOGGER=${LOGGER:-wandb}

if [ ! -x "${ISAAC_PYTHON}" ]; then
  echo "Expected Isaac Sim launcher at ${ISAAC_PYTHON}, but it was not found or not executable." >&2
  exit 1
fi
if [ ! -f "${TRAIN_PY}" ]; then
  echo "Training entry point not found at ${TRAIN_PY}. Adjust SAFE_LOCO_ROOT or TRAIN_PY." >&2
  exit 1
fi

if [ -z "${LOW_LEVEL_POLICY_PATH:-}" ]; then
  echo "LOW_LEVEL_POLICY_PATH is not set. Provide the exported low-level locomotion policy (.pt)." >&2
  exit 1
fi
if [ ! -f "${LOW_LEVEL_POLICY_PATH}" ]; then
  echo "Low-level policy checkpoint not found at ${LOW_LEVEL_POLICY_PATH}." >&2
  exit 1
fi

TRAIN_CMD=(
  "${ISAAC_PYTHON}"
  "${TRAIN_PY}"
  --headless
  --task "${TASK}"
  --algo "${ALGO}"
  --run_name "${RUN_NAME}"
  --logger "${LOGGER}"
  "env.actions.pre_trained_policy_action.policy_path=${LOW_LEVEL_POLICY_PATH}"
)

if [ -n "${NUM_ENVS:-}" ]; then
  TRAIN_CMD+=(--num_envs "${NUM_ENVS}")
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then
  TRAIN_CMD+=(--max_iterations "${MAX_ITERATIONS}")
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
  TRAIN_CMD+=(--log_project_name "${WANDB_PROJECT}")
fi
if [ -n "${SEED:-}" ]; then
  TRAIN_CMD+=(--seed "${SEED}")
fi
if [ -n "${TRAIN_ADDITIONAL_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  TRAIN_CMD+=(${TRAIN_ADDITIONAL_ARGS})
fi

echo "=========================================="
echo "Launching ${ALGO} navigation training"
echo "Task: ${TASK}"
echo "Run name: ${RUN_NAME}"
echo "Low-level policy: ${LOW_LEVEL_POLICY_PATH}"
echo "Isaac Sim Python: ${ISAAC_PYTHON}"
echo "=========================================="

srun --ntasks=1 "${TRAIN_CMD[@]}"
