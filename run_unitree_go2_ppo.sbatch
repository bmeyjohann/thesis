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

mkdir -p logs wandb

export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-unitree_ros/unitree_ros}"
export GIT_PYTHON_REFRESH=quiet
unset CUDA_VISIBLE_DEVICES

source "sc_venv_template_isaaclab/activate.sh"
source ".venv/bin/activate"

export XDG_CACHE_HOME=/p/project1/hai_1074/meyjohann1/apptainer/cache \
export XDG_DATA_HOME=/p/project1/hai_1074/meyjohann1/apptainer/data \
export KIT_USER_ROOT=/p/project1/hai_1074/meyjohann1/apptainer/kit \
export KIT_USER_DATA=/p/project1/hai_1074/meyjohann1/apptainer/data \
export KIT_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache \
export KIT_DERIVED_DATA_CACHE=/p/project1/hai_1074/meyjohann1/apptainer/DerivedDataCache \
export KIT_PIP_INSTALL_PATH=/p/project1/hai_1074/meyjohann1/apptainer/pip3-envs \
export OMNI_USER_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache \
export OMNI_KIT_USER_DIR=/p/project1/hai_1074/meyjohann1/apptainer/kit \
export OMNI_KIT_DATA_DIR=/p/project1/hai_1074/meyjohann1/apptainer/data \
export OMNI_KIT_CACHE_DIR=/p/project1/hai_1074/meyjohann1/apptainer/cache

export OMNI_DISABLE_EXTENSIONS="omni.iray.libs,omni.mdl.neuraylib,omni.kit.usd.mdl,carb.scenerenderer-rtx.plugin,omni.rtx.*,omni.hydra.rtx"

export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-wandb}"
export WANDB_START_METHOD=thread

TASK=${TASK:-New-Isaac-Velocity-Flat-Unitree-Go2-v0}
ALGO=${ALGO:-ppo}
RUN_NAME=${RUN_NAME:-simple_reward_config_ppo_${SLURM_JOB_ID}}
CONFIG_DIR=${CONFIG_DIR:-scripts/configs}
CONFIG_NAME=${CONFIG_NAME:-simple_reward_config}
LOGGER=${LOGGER:-wandb}
ISAAC_PYTHON=${ISAAC_PYTHON:-~/meyjohann1/isaacsim5.1/python.sh}

if [ ! -x "${ISAAC_PYTHON}" ]; then
  echo "Expected Isaac Sim launcher at ${ISAAC_PYTHON}, but it was not found or not executable." >&2
  exit 1
fi

TRAIN_CMD=(
  "${ISAAC_PYTHON}"
  "safe-locomotion/scripts/rsl_rl/train.py"
  --headless
  --task "${TASK}"
  --algo "${ALGO}"
  --run_name "${RUN_NAME}"
  --config-dir "${CONFIG_DIR}"
  --config-name "${CONFIG_NAME}"
  --logger "${LOGGER}"
)

if [ -n "${TRAIN_ADDITIONAL_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  TRAIN_CMD+=(${TRAIN_ADDITIONAL_ARGS})
fi

echo "Launching ${ALGO} training for ${TASK} with run name ${RUN_NAME}"
echo "Using Isaac Sim Python: ${ISAAC_PYTHON}"

srun --ntasks=1 "${TRAIN_CMD[@]}"
