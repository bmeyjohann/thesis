#!/bin/bash
#SBATCH --job-name=isaacsmoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --partition=booster
#SBATCH --account=<your_account>
#SBATCH --output=logs/isaacsmoke_%j.out
#SBATCH --error=logs/isaacsmoke_%j.err

# Basic Isaac Sim + Isaac Lab + FastTD3 smoke test executed inside an Apptainer image.
# Adjust the SBATCH directives (partition, account, resources) as needed for JUWELS.

set -euo pipefail

IMAGE_PATH="${IMAGE_PATH:-$(realpath ../apptainer/images/isaacsim-5.0.0.sif)}"
WORKSPACE_ROOT="$(realpath "$(pwd)")"
LOG_DIR="${WORKSPACE_ROOT}/logs"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Expected Apptainer image at ${IMAGE_PATH} but it was not found."
    exit 1
fi

mkdir -p "${LOG_DIR}" models

# Forward repo into the container at /workspace and make fasttd3 importable.
export APPTAINERENV_PYTHONPATH="/workspace"
export APPTAINERENV_WANDB_MODE="offline"
export APPTAINERENV_WANDB_CONSOLE="off"
export APPTAINERENV_WANDB_SILENT="true"
export APPTAINERENV_TORCHDYNAMO_INLINE_INBUILT_NN_MODULES="1"
export APPTAINERENV_MUJOCO_GL="egl"
export APPTAINERENV_ACCEPT_EULA="Y"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA="yes"

APPTAINER_BASE_CMD=("apptainer" "exec" "--nv" "--bind" "${WORKSPACE_ROOT}:/workspace" "${IMAGE_PATH}")

echo "[$(date)] Verifying Isaac Sim container imports..."
"${APPTAINER_BASE_CMD[@]}" /isaac-sim/python.sh -c "import isaacsim, isaaclab, torch; print('Imports OK')"

echo "[$(date)] Ensuring container venv is present..."
"${APPTAINER_BASE_CMD[@]}" /bin/bash -lc 'cd /workspace && if [[ ! -d sc_venv_template_isaaclab/venv ]]; then bash sc_venv_template_isaaclab/setup.sh; else echo "Using existing venv at sc_venv_template_isaaclab/venv"; fi'

echo "[$(date)] Launching FastTD3 IsaacLab smoke training..."
JOB_SEED="${SLURM_JOB_ID:-0}"
"${APPTAINER_BASE_CMD[@]}" env JOB_SEED="${JOB_SEED}" /bin/bash -lc '
set -euo pipefail
cd /workspace
source sc_venv_template_isaaclab/activate.sh
python fasttd3/fast_sac/train.py \
    --env_name Isaac-Lift-Cube-Franka-v0 \
    --total_timesteps 2048 \
    --num_envs 16 \
    --num_eval_envs 16 \
    --seed ${JOB_SEED} \
    --no-use-wandb \
    --no-compile \
    --no-amp \
    --exp_name isaaclab_smoke \
    --project smoke-tests \
    --eval_interval 2048 \
    --save_interval 2048 \
    --measure_burnin 0
'

TRAIN_EXIT_CODE=$?

if [[ ${TRAIN_EXIT_CODE} -ne 0 ]]; then
    echo "[$(date)] FastTD3 smoke training failed with exit code ${TRAIN_EXIT_CODE}."
    exit ${TRAIN_EXIT_CODE}
fi

echo "[$(date)] FastTD3 smoke training completed successfully."
