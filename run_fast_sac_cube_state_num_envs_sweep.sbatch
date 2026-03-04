#!/bin/bash
# Sweep num_envs for FastSAC cube state runs (one run per GPU).
# Focus: throughput scaling (Perf/total_fps and env_steps per wall time).
#
# Example:
#   bash submit_job.sh run_fast_sac_cube_state_num_envs_sweep.sbatch
#   NUM_ENVS_SWEEP="32 64 96 128" bash submit_job.sh run_fast_sac_cube_state_num_envs_sweep.sbatch

#SBATCH --job-name=fastsac_cube_nenv_sweep
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_cube_nenv_sweep_%j.out
#SBATCH --error=logs/fastsac_cube_nenv_sweep_%j.err

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

ENV_ID=${ENV_ID:-cube-triple-singletask-task5-v0}
NUM_ENVS_SWEEP=${NUM_ENVS_SWEEP:-"32 64 96 128"}
TOTAL_STEPS=${TOTAL_STEPS:-120000}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
LOG_INTERVAL=${LOG_INTERVAL:-2000}
PROJECT=${PROJECT:-ogbench_fastsac_cube_stack_num_envs_sweep}
REWARD_TYPE=${REWARD_TYPE:-sparse}
STEP_PENALTY=${STEP_PENALTY:-0.0}
TOLERANCE_TYPE=${TOLERANCE_TYPE:-l2}
TOLERANCE_VALUE=${TOLERANCE_VALUE:-0.05}

BASE_ARGS="--obs_mode state \
--total_timesteps ${TOTAL_STEPS} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
--reward_type ${REWARD_TYPE} --step_penalty ${STEP_PENALTY} \
--use_wandb --project ${PROJECT} \
--use_intervention --intervention_mode agent --teacher_type cube_plan \
--teacher_target_mode sequential --cube_success_tolerance 0.04 \
--tolerance_type ${TOLERANCE_TYPE} --tolerance_value ${TOLERANCE_VALUE} \
--intervention_episode_prob 1.0 --intervention_episode_prob_min 1.0 \
--intervention_episode_prob_decay_steps 0 --intervention_episode_prob_decay_start 0 \
--pref_buffer_enable --pref_capacity 200000 --pref_sample_ratio 0.3 \
--pref_rank_margin 0.1 --pref_rank_weight 1.0"

launch_nenv_run () {
  local GPU_ID=$1
  local NENV=$2

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_fast_sac_ogbench_manip.py \
    --env_name "${ENV_ID}" \
    --num_envs "${NENV}" \
    ${BASE_ARGS} \
    --exp_name "fastsac_own_${ENV_ID}_nenv_${NENV}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/fastsac_own_${ENV_ID}_nenv_${NENV}_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
  echo "Started nenv=${NENV} on GPU ${GPU_ID}"
}

echo "Launching cube num_envs sweep on ${ENV_ID}"
echo "NUM_ENVS_SWEEP=${NUM_ENVS_SWEEP}"

GPU_IDS=(0 1 2 3)
IDX=0
PIDS=()
for NENV in ${NUM_ENVS_SWEEP}; do
  GPU_ID=${GPU_IDS[$IDX]}
  launch_nenv_run "${GPU_ID}" "${NENV}"
  IDX=$(( (IDX + 1) % ${#GPU_IDS[@]} ))
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more cube num_envs runs failed."
  exit 1
fi

echo "Cube num_envs sweep completed successfully."
