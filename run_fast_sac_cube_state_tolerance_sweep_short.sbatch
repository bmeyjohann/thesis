#!/bin/bash
#SBATCH --job-name=fastsac_cube_tol_sweep
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_cube_tol_sweep_%j.out
#SBATCH --error=logs/fastsac_cube_tol_sweep_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SLURM_TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}}
mkdir -p "${XDG_CACHE_HOME}"
export WANDB_DIR=${WANDB_DIR:-$PWD}
mkdir -p "${WANDB_DIR}/wandb"

ENV_ID=${ENV_ID:-cube-double-v0}
TOTAL_STEPS=${TOTAL_STEPS:-250000}
NUM_ENVS=${NUM_ENVS:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
PROJECT=${PROJECT:-ogbench_fastsac_cube_tol_sweep_short}
REWARD_TYPE=${REWARD_TYPE:-dense}
STEP_PENALTY=${STEP_PENALTY:-0.0}
TOLERANCE_TYPE=${TOLERANCE_TYPE:-l2}
TOLERANCES=${TOLERANCES:-"0.10 0.20 0.30 0.50"}

BASE_ARGS="--obs_mode state \
--total_timesteps ${TOTAL_STEPS} --num_envs ${NUM_ENVS} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
--reward_type ${REWARD_TYPE} --step_penalty ${STEP_PENALTY} \
--use_wandb --project ${PROJECT} \
--use_intervention --intervention_mode agent --teacher_type cube_plan \
--intervention_episode_prob 1.0 --intervention_episode_prob_min 1.0 \
--intervention_episode_prob_decay_steps 0 --intervention_episode_prob_decay_start 0 \
--pref_buffer_enable --pref_capacity 200000 --pref_sample_ratio 0.3 \
--pref_rank_margin 0.1 --pref_rank_weight 1.0 \
--viz_on_checkpoint --viz_grid_resolution 32 --viz_quiver_stride 2"

launch_tol_run () {
  local GPU_ID=$1
  local TOL=$2
  local TOL_TAG
  TOL_TAG=$(echo "${TOL}" | tr '.' 'p')

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_fast_sac_ogbench.py \
    --env_name "${ENV_ID}" \
    ${BASE_ARGS} \
    --tolerance_type "${TOLERANCE_TYPE}" --tolerance_value "${TOL}" \
    --exp_name "fastsac_own_${ENV_ID}_tol_${TOL_TAG}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/fastsac_own_${ENV_ID}_tol_${TOL_TAG}_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
}

echo "Launching short FastSAC tolerance sweep on ${ENV_ID}"

PIDS=()
GPU_IDS=(0 1 2 3)
IDX=0
for TOL in ${TOLERANCES}; do
  GPU_ID=${GPU_IDS[$IDX]}
  launch_tol_run "${GPU_ID}" "${TOL}"
  IDX=$(( (IDX + 1) % ${#GPU_IDS[@]} ))
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more FastSAC tolerance sweep runs failed."
  exit 1
fi

echo "FastSAC tolerance sweep completed successfully."
