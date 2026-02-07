#!/bin/bash
#SBATCH --job-name=fastsac_cube_state_own
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_cube_state_own_%j.out
#SBATCH --error=logs/fastsac_cube_state_own_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SLURM_TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}}
mkdir -p "${XDG_CACHE_HOME}"
export WANDB_DIR=${WANDB_DIR:-$PWD/wandb}
mkdir -p "${WANDB_DIR}"

ENV_IDS=${ENV_IDS:-"cube-single-v0 cube-double-v0 cube-triple-v0 cube-quadruple-v0 cube-octuple-v0"}
TOTAL_STEPS=${TOTAL_STEPS:-12000000}
NUM_ENVS=${NUM_ENVS:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
LOG_INTERVAL=${LOG_INTERVAL:-100000}
PROJECT=${PROJECT:-ogbench_fast_sac_cube_state_own}
REWARD_TYPE=${REWARD_TYPE:-dense}
STEP_PENALTY=${STEP_PENALTY:-0.0}
TOLERANCE_TYPE=${TOLERANCE_TYPE:-l2}
TOLERANCE_VALUE=${TOLERANCE_VALUE:-0.3}

BASE_ARGS="--obs_mode state \
--total_timesteps ${TOTAL_STEPS} --num_envs ${NUM_ENVS} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
--reward_type ${REWARD_TYPE} --step_penalty ${STEP_PENALTY} \
--use_wandb --project ${PROJECT} \
--use_intervention --intervention_mode agent --teacher_type cube_plan \
--tolerance_type ${TOLERANCE_TYPE} --tolerance_value ${TOLERANCE_VALUE} \
--pref_buffer_enable --pref_capacity 200000 --pref_sample_ratio 0.3 \
--pref_rank_margin 0.1 --pref_rank_weight 1.0 \
--viz_on_checkpoint --viz_grid_resolution 32 --viz_quiver_stride 2"

echo "Launching FastSAC cube state OWN runs for: ${ENV_IDS}"

GPU_IDS=(0 1 2 3)
GPU_INDEX=0
PIDS=()

for ENV_ID in ${ENV_IDS}; do
  GPU_ID=${GPU_IDS[$GPU_INDEX]}
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_fast_sac_ogbench.py \
    --env_name "${ENV_ID}" \
    --exp_name "fastsac_own_${ENV_ID}_state_${SLURM_JOB_ID}" \
    ${BASE_ARGS} \
    2>&1 | tee "logs/fastsac_own_${ENV_ID}_state_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
  GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPU_IDS[@]} ))
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more FastSAC cube state OWN runs failed."
  exit 1
fi

echo "FastSAC cube state OWN runs completed successfully."
