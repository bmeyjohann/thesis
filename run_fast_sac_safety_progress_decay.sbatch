#!/bin/bash
#SBATCH --job-name=fastsac_safety_progress
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_safety_progress_%j.out
#SBATCH --error=logs/fastsac_safety_progress_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

ENV_A=${ENV_A:-pointmaze-arena-v0}
ENV_B=${ENV_B:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-20000000}
NUM_ENVS=${NUM_ENVS:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
LOG_INTERVAL=${LOG_INTERVAL:-100000}
PROJECT=${PROJECT:-ogbench_fast_sac_safety_progress}
STEP_PENALTY=${STEP_PENALTY:-0.001}

INTERVENTION_ENABLE_AFTER_STEPS=${INTERVENTION_ENABLE_AFTER_STEPS:-10}
INTERVENTION_EP_PROB=${INTERVENTION_EP_PROB:-0.5}
INTERVENTION_EP_PROB_MIN=${INTERVENTION_EP_PROB_MIN:-0.1}
INTERVENTION_DECAY_START=${INTERVENTION_DECAY_START:-500}
INTERVENTION_DECAY_STEPS=${INTERVENTION_DECAY_STEPS:-$((TOTAL_STEPS / NUM_ENVS))}

VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-500000}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}

BASE_ARGS="--obs_mode state --reward_type sparse --dense_reward_scale 0.0 --step_penalty ${STEP_PENALTY} \
--total_timesteps ${TOTAL_STEPS} --num_envs ${NUM_ENVS} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
--use_wandb --project ${PROJECT} \
--use_intervention --intervention_mode agent_safety_progress --teacher_type bfs \
--tolerance_type angle --tolerance_value 30 \
--intervention_safety_margin_frac 0.25 --intervention_release_steps 3 \
--intervention_enable_after_steps ${INTERVENTION_ENABLE_AFTER_STEPS} \
--intervention_episode_prob ${INTERVENTION_EP_PROB} \
--intervention_episode_prob_min ${INTERVENTION_EP_PROB_MIN} \
--intervention_episode_prob_decay_start ${INTERVENTION_DECAY_START} \
--intervention_episode_prob_decay_steps ${INTERVENTION_DECAY_STEPS} \
--no_hard_block_lethal \
--viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
--viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE}"

echo "Launching FastSAC safety_progress runs on ${ENV_A} and ${ENV_B}"

echo "Intervention schedule: p0=${INTERVENTION_EP_PROB} -> pmin=${INTERVENTION_EP_PROB_MIN} over ${INTERVENTION_DECAY_STEPS} steps per env (start ${INTERVENTION_DECAY_START})"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench_maze.py \
  --env_name ${ENV_A} \
  --exp_name "fastsac_${ENV_A}_safety_progress_${SLURM_JOB_ID}" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${ENV_A}_fastsac_safety_progress_${SLURM_JOB_ID}.log &
PID_A=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench_maze.py \
  --env_name ${ENV_B} \
  --exp_name "fastsac_${ENV_B}_safety_progress_${SLURM_JOB_ID}" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${ENV_B}_fastsac_safety_progress_${SLURM_JOB_ID}.log &
PID_B=$!

echo "PIDs: $PID_A (arena)  $PID_B (danger)"

wait $PID_A; EXIT_A=$?
wait $PID_B; EXIT_B=$?

if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ]; then
  echo "One or more FastSAC runs failed ($EXIT_A, $EXIT_B)"
  exit 1
fi

echo "FastSAC safety_progress runs completed successfully."
