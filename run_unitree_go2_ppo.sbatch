#!/bin/bash
#SBATCH --job-name=unitree_go2_ppo
#SBATCH --account=<your_account>
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/unitree_go2_ppo_%j.out
#SBATCH --error=logs/unitree_go2_ppo_%j.err

set -euo pipefail

# SLURM automatically sets SLURM_SUBMIT_DIR to where you ran sbatch
REPO_ROOT="${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Working directory: ${REPO_ROOT}"
echo "=========================================="

cd "${REPO_ROOT}"

# Create log directories
mkdir -p logs wandb

# Environment setup matching the README
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-${REPO_ROOT}/unitree_ros/unitree_ros}"
export GIT_PYTHON_REFRESH=quiet
unset CUDA_VISIBLE_DEVICES

# Activate container-focused env (Python deps, PYTHONPATH)
source "${REPO_ROOT}/sc_venv_template_isaaclab_container/activate.sh"

ISAAC_ROOT=${ISAAC_ROOT:-/p/project1/hai_1074/meyjohann1/apptainer}
APPTAINER_IMG=${APPTAINER_IMG:-${ISAAC_ROOT}/images/isaacsim-5.0.0.sif}
KIT_USER_ROOT=${KIT_USER_ROOT:-${ISAAC_ROOT}/.kit}
KIT_USER_DATA=${KIT_USER_DATA:-${KIT_USER_ROOT}/data}
KIT_CACHE_DIR=${KIT_CACHE_DIR:-${KIT_USER_ROOT}/cache}
KIT_DERIVED_DATA_CACHE=${KIT_DERIVED_DATA_CACHE:-${KIT_USER_ROOT}/DerivedDataCache}
KIT_PIP_INSTALL_PATH=${KIT_PIP_INSTALL_PATH:-${KIT_USER_ROOT}/pip3-envs}
OVERLAY=${OVERLAY:-${ISAAC_ROOT}/overlay}

mkdir -p "${KIT_USER_ROOT}"/{cache,DerivedDataCache,data,pip3-envs,cache/nv_shadercache}
mkdir -p "${ISAAC_ROOT}/isaac-cache"/{ov,matplotlib,fontconfig,nvidia,warp}
mkdir -p "${OVERLAY}/py310/omni"/{cache,data}
mkdir -p "${OVERLAY}/py311/omni"/{cache,data}

if [ ! -f "${APPTAINER_IMG}" ]; then
  echo "Apptainer image not found at ${APPTAINER_IMG}" >&2
  exit 1
fi

export WANDB_MODE=offline
export WANDB_DIR="${REPO_ROOT}/wandb"
export WANDB_START_METHOD=thread

TASK=${TASK:-New-Isaac-Velocity-Flat-Unitree-Go2-v0}
ALGO=${ALGO:-ppo}
RUN_NAME=${RUN_NAME:-simple_reward_config_ppo_${SLURM_JOB_ID}}
LOGGER=${LOGGER:-wandb}

BIND_ARGS=(
  --bind "${KIT_USER_ROOT}:/workspace/.kit:rw"
  --bind "${ISAAC_ROOT}/isaac-cache/ov:/root/.cache/ov:rw"
  --bind "${ISAAC_ROOT}/isaac-cache/matplotlib:/root/.cache/matplotlib:rw"
  --bind "${ISAAC_ROOT}/isaac-cache/fontconfig:/root/.cache/fontconfig:rw"
  --bind "${ISAAC_ROOT}/isaac-cache/nvidia:/root/.cache/nvidia:rw"
  --bind "${ISAAC_ROOT}/isaac-cache/warp:/root/.cache/warp:rw"
  --bind "${OVERLAY}/py310/omni:/workspace/project/env/lib/python3.10/site-packages/omni:rw"
  --bind "${OVERLAY}/py311/omni:/workspace/project/env/lib/python3.11/site-packages/omni:rw"
)

ENV_ARGS=(
  --env OMNI_ENV_PRIVACY_CONSENT=1
  --env OMNI_ACCEPT_EULA=Y
  --env OMNI_KIT_ACCEPT_EULA=Y
  --env ACCEPT_EULA=Y
  --env OMNI_SERVER=
  --env XDG_CACHE_HOME=/workspace/.kit/cache
  --env XDG_DATA_HOME=/workspace/.kit/data
  --env KIT_USER_ROOT=/workspace/.kit
  --env KIT_USER_DATA=/workspace/.kit/data
  --env KIT_CACHE_DIR=/workspace/.kit/cache
  --env KIT_DERIVED_DATA_CACHE=/workspace/.kit/DerivedDataCache
  --env KIT_PIP_INSTALL_PATH=/workspace/.kit/pip3-envs
  --env OMNI_USER_CACHE_DIR=/workspace/.kit/cache
  --env OMNI_KIT_USER_DIR=/workspace/.kit
  --env OMNI_KIT_DATA_DIR=/workspace/.kit/data
  --env OMNI_KIT_CACHE_DIR=/workspace/.kit/cache
  --env OMNI_KIT_ARGS="--portable-root=/workspace/.kit --/app/dataDir=/workspace/.kit/data --/app/cacheDir=/workspace/.kit/cache --/app/derivedData/cachePath=/workspace/.kit/DerivedDataCache --/rtx/shaderdb/cachePath=/workspace/.kit/cache/nv_shadercache --/exts/omni.kit.pipapi/enable=0"
  --env WANDB_MODE=offline
  --env WANDB_DIR="${WANDB_DIR}"
  --env WANDB_START_METHOD=thread
)

TRAIN_ARGS=(
  --headless
  --task "${TASK}"
  --algo "${ALGO}"
  --run_name "${RUN_NAME}"
  --logger "${LOGGER}"
)

if [ -n "${NUM_ENVS:-}" ]; then
  TRAIN_ARGS+=(--num_envs "${NUM_ENVS}")
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then
  TRAIN_ARGS+=(--max_iterations "${MAX_ITERATIONS}")
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
  TRAIN_ARGS+=(--log_project_name "${WANDB_PROJECT}")
fi
if [ -n "${SEED:-}" ]; then
  TRAIN_ARGS+=(--seed "${SEED}")
fi
if [ -n "${TRAIN_ADDITIONAL_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  TRAIN_ARGS+=(${TRAIN_ADDITIONAL_ARGS})
fi

APPTAINER_BASE_ARGS=(--nv --cleanenv --writable-tmpfs --pwd "${REPO_ROOT}")

APPTAINER_CMD=(
  apptainer exec
  "${APPTAINER_BASE_ARGS[@]}"
  "${BIND_ARGS[@]}"
  "${ENV_ARGS[@]}"
  "${APPTAINER_IMG}"
  IsaacLab/isaaclab.sh
  -p "${REPO_ROOT}/safe-locomotion/scripts/rsl_rl/train.py"
  "${TRAIN_ARGS[@]}"
)

echo "=========================================="
echo "Launching ${ALGO} training inside ${APPTAINER_IMG}"
echo "Task: ${TASK}"
echo "Run name: ${RUN_NAME}"
echo "=========================================="

srun --ntasks=1 "${APPTAINER_CMD[@]}"
