#!/bin/bash
#SBATCH --job-name=fastsac_highway_expert
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_highway_expert_%j.out
#SBATCH --error=logs/fastsac_highway_expert_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

ENV_ID=${ENV_ID:-highway-v0}
TOTAL_STEPS=${TOTAL_STEPS:-20000000}
NUM_ENVS=${NUM_ENVS:-32}
EVAL_ENVS=${EVAL_ENVS:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}
LOG_INTERVAL=${LOG_INTERVAL:-100000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500000}
PROJECT=${PROJECT:-fastsac_highway_expert}

WANDB_MODE=offline python -u train_fast_sac_highway.py \
  --env_name ${ENV_ID} \
  --exp_name "fastsac_highway_${SLURM_JOB_ID}" \
  --total_timesteps ${TOTAL_STEPS} \
  --num_envs ${NUM_ENVS} \
  --num_eval_envs ${EVAL_ENVS} \
  --save_interval ${SAVE_INTERVAL} \
  --log_interval ${LOG_INTERVAL} \
  --eval_interval ${EVAL_INTERVAL} \
  --offroad_terminal \
  --offroad_penalty -1.0 \
  --use_wandb \
  --project ${PROJECT} \
  2>&1 | tee logs/${ENV_ID}_fastsac_expert_${SLURM_JOB_ID}.log
