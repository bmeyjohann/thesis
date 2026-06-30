#!/bin/bash
#SBATCH --job-name=unitree_nav_eval
#SBATCH --account=hai_1074
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/unitree_nav_eval_%j.out
#SBATCH --error=logs/unitree_nav_eval_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SAFE_LOCO_ROOT=${SAFE_LOCO_ROOT:-${REPO_ROOT}/safe-locomotion}
cd "${REPO_ROOT}"
mkdir -p logs wandb

export TERM=xterm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
source "${REPO_ROOT}/sc_venv_template_isaaclab_container/activate.sh"

ISAAC_ROOT=${ISAAC_ROOT:-/p/project1/hai_1074/meyjohann1/apptainer}
APPTAINER_IMG=${APPTAINER_IMG:-${ISAAC_ROOT}/images/isaacsim-5.0.0.sif}
KIT_USER_ROOT=${KIT_USER_ROOT:-${ISAAC_ROOT}/.kit}
OVERLAY=${OVERLAY:-${ISAAC_ROOT}/overlay}

mkdir -p "${KIT_USER_ROOT}"/{cache,DerivedDataCache,data,pip3-envs,cache/nv_shadercache}
mkdir -p "${ISAAC_ROOT}/isaac-cache"/{ov,matplotlib,fontconfig,nvidia,warp}
mkdir -p "${OVERLAY}/py310/omni"/{cache,data}
mkdir -p "${OVERLAY}/py311/omni"/{cache,data}

TASK=${TASK:-Isaac-Navigation-NoObstacles-Flat-Go2-v0}
LOW_LEVEL_POLICY_PATH=${LOW_LEVEL_POLICY_PATH:-${REPO_ROOT}/logs/rsl_rl/unitree_go2_flat/2025-11-04_23-55-26_simple_reward_config_ppo_12808330/exported/policy.pt}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/logs/isaac_navigation_scripted/no_obstacle_${SLURM_JOB_ID}}
NUM_EPISODES=${NUM_EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-300}
VIDEO_LENGTH=${VIDEO_LENGTH:-350}
SEED=${SEED:-0}

if [ ! -f "${LOW_LEVEL_POLICY_PATH}" ]; then
  echo "Low-level policy not found: ${LOW_LEVEL_POLICY_PATH}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

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
  --env WANDB_DIR="${REPO_ROOT}/wandb"
  --env WANDB_START_METHOD=thread
  --env MPLCONFIGDIR="${MPLCONFIGDIR}"
  --env TERM=xterm
  --env GIT_PYTHON_REFRESH=quiet
)

CMD=(
  apptainer exec
  --nv --cleanenv --writable-tmpfs --pwd "${REPO_ROOT}"
  "${BIND_ARGS[@]}"
  "${ENV_ARGS[@]}"
  "${APPTAINER_IMG}"
  IsaacLab/isaaclab.sh
  -p "${SAFE_LOCO_ROOT}/scripts/rsl_rl/eval_navigation_scripted.py"
  --headless
  --video
  --task "${TASK}"
  --low_level_policy_path "${LOW_LEVEL_POLICY_PATH}"
  --num_envs 1
  --num_episodes "${NUM_EPISODES}"
  --max_steps "${MAX_STEPS}"
  --video_length "${VIDEO_LENGTH}"
  --output_dir "${OUT_DIR}"
  --seed "${SEED}"
  --stop_on_success
)

echo "Scripted navigation eval"
echo "task: ${TASK}"
echo "low-level: ${LOW_LEVEL_POLICY_PATH}"
echo "out: ${OUT_DIR}"
(
  srun --ntasks=1 "${CMD[@]}" 2>&1
) | tee "logs/unitree_nav_eval_${SLURM_JOB_ID}.eval.log"
