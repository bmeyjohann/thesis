#!/bin/bash
#SBATCH --job-name=fastsac_cube_state_cmp
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/fastsac_cube_state_cmp_%j.out
#SBATCH --error=logs/fastsac_cube_state_cmp_%j.err

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
PROJECT=${PROJECT:-ogbench_fast_sac_cube_state_compare}
REWARD_TYPE=${REWARD_TYPE:-dense}
STEP_PENALTY=${STEP_PENALTY:-0.0}
TOLERANCE_TYPE=${TOLERANCE_TYPE:-l2}
TOLERANCE_VALUE=${TOLERANCE_VALUE:-0.3}
DEMO_PREFILL_STEPS=${DEMO_PREFILL_STEPS:-500000}
DEMO_BUFFER_CAPACITY=${DEMO_BUFFER_CAPACITY:-500000}
DEMO_SAMPLE_RATIO=${DEMO_SAMPLE_RATIO:-0.5}

BASE_ARGS="--obs_mode state \
--total_timesteps ${TOTAL_STEPS} --num_envs ${NUM_ENVS} \
--save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
--reward_type ${REWARD_TYPE} --step_penalty ${STEP_PENALTY} \
--use_wandb --project ${PROJECT} \
--use_intervention --intervention_mode agent --teacher_type cube_plan \
--tolerance_type ${TOLERANCE_TYPE} --tolerance_value ${TOLERANCE_VALUE} \
--viz_on_checkpoint --viz_grid_resolution 32 --viz_quiver_stride 2"

OURS_ARGS="--pref_buffer_enable --pref_capacity 200000 --pref_sample_ratio 0.3 \
--pref_rank_margin 0.1 --pref_rank_weight 1.0"

BASELINE_ARGS="--demo_buffer_enable --demo_buffer_capacity ${DEMO_BUFFER_CAPACITY} \
--demo_sample_ratio ${DEMO_SAMPLE_RATIO} --demo_prefill_steps ${DEMO_PREFILL_STEPS} \
--demo_prefill_target demo --demo_prefill_intervention_mode agent \
--demo_prefill_episode_prob 1.0"

echo "Launching SLOW comparison (ours vs demo-buffer baseline) for: ${ENV_IDS}"

for ENV_ID in ${ENV_IDS}; do
  echo "Starting env=${ENV_ID}"

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_fast_sac_ogbench.py \
    --env_name "${ENV_ID}" \
    --exp_name "fastsac_own_${ENV_ID}_state_${SLURM_JOB_ID}" \
    ${BASE_ARGS} \
    ${OURS_ARGS} \
    2>&1 | tee "logs/fastsac_own_${ENV_ID}_state_${SLURM_JOB_ID}.log" &
  PID_OURS=$!

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_fast_sac_ogbench.py \
    --env_name "${ENV_ID}" \
    --exp_name "fastsac_baseline_${ENV_ID}_state_${SLURM_JOB_ID}" \
    ${BASE_ARGS} \
    ${BASELINE_ARGS} \
    2>&1 | tee "logs/fastsac_baseline_${ENV_ID}_state_${SLURM_JOB_ID}.log" &
  PID_BASE=$!

  wait "${PID_OURS}"; EXIT_OURS=$?
  wait "${PID_BASE}"; EXIT_BASE=$?

  if [ "${EXIT_OURS}" -ne 0 ] || [ "${EXIT_BASE}" -ne 0 ]; then
    echo "Comparison failed for ${ENV_ID} (ours=${EXIT_OURS}, baseline=${EXIT_BASE})"
    exit 1
  fi
done

echo "Slow FastSAC cube state comparison completed successfully."
