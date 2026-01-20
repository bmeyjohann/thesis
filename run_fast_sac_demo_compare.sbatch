#!/bin/bash
#SBATCH --job-name=fastsac_demo_compare
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_demo_compare_%j.out
#SBATCH --error=logs/fastsac_demo_compare_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

ENV_ID=${ENV_ID:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-20000000}
NUM_ENVS=${NUM_ENVS:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
LOG_INTERVAL=${LOG_INTERVAL:-100000}
PROJECT=${PROJECT:-ogbench_fast_sac_demo_compare}
STEP_PENALTY=${STEP_PENALTY:-0.001}

DEMO_PREFILL_STEPS=${DEMO_PREFILL_STEPS:-2000000}
DEMO_BUFFER_CAPACITY=${DEMO_BUFFER_CAPACITY:-2000000}
DEMO_SAMPLE_RATIO=${DEMO_SAMPLE_RATIO:-0.5}

VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-500000}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}

BASE_ARGS="--obs_mode state --reward_type sparse --dense_reward_scale 0.0 --step_penalty ${STEP_PENALTY} \
--total_timesteps ${TOTAL_STEPS} --num_envs ${NUM_ENVS} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
--use_wandb --project ${PROJECT} \
--no_hard_block_lethal \
--viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
--viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE}"

echo "Launching FastSAC demo comparison runs on ${ENV_ID}"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py \
  --env_name ${ENV_ID} \
  --exp_name "fastsac_${ENV_ID}_prefill_replay_${SLURM_JOB_ID}" \
  --demo_prefill_steps ${DEMO_PREFILL_STEPS} \
  --demo_prefill_target replay \
  --demo_prefill_intervention_mode agent_safety_progress \
  --demo_prefill_episode_prob 1.0 \
  ${BASE_ARGS} \
  2>&1 | tee logs/${ENV_ID}_fastsac_prefill_replay_${SLURM_JOB_ID}.log &
PID_A=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py \
  --env_name ${ENV_ID} \
  --exp_name "fastsac_${ENV_ID}_demo_buffer_${SLURM_JOB_ID}" \
  --demo_buffer_enable --demo_buffer_capacity ${DEMO_BUFFER_CAPACITY} \
  --demo_sample_ratio ${DEMO_SAMPLE_RATIO} \
  --demo_prefill_steps ${DEMO_PREFILL_STEPS} \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent_safety_progress \
  --demo_prefill_episode_prob 1.0 \
  ${BASE_ARGS} \
  2>&1 | tee logs/${ENV_ID}_fastsac_demo_buffer_${SLURM_JOB_ID}.log &
PID_B=$!

echo "PIDs: $PID_A (prefill replay)  $PID_B (demo buffer)"

wait $PID_A; EXIT_A=$?
wait $PID_B; EXIT_B=$?

if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ]; then
  echo "One or more FastSAC runs failed ($EXIT_A, $EXIT_B)"
  exit 1
fi

echo "FastSAC demo comparison runs completed successfully."
