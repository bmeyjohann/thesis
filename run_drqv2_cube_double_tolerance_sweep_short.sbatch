#!/bin/bash
#SBATCH --job-name=drqv2_cube_tol_sweep
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_cube_tol_sweep_%j.out
#SBATCH --error=logs/drqv2_cube_tol_sweep_%j.err

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

OGBENCH_ENV=${OGBENCH_ENV:-visual-cube-double-v0}
TOTAL_STEPS=${TOTAL_STEPS:-250000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
EVAL_EVERY=${EVAL_EVERY:-50000}
PROJECT=${PROJECT:-ogbench_cube_double_tol_sweep_short}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-1}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TEACHER_TYPE=${TEACHER_TYPE:-cube_plan}
REWARD_TYPE=${REWARD_TYPE:-dense}
TOLERANCE_TYPE=${TOLERANCE_TYPE:-l2}
TOLERANCES=${TOLERANCES:-"0.10 0.20 0.30 0.50"}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--nstep 1 \
--use_wandb --viz_on_checkpoint \
--pref_buffer_enable --pref_capacity 200000 --pref_chunk_size 128 \
--pref_sample_ratio 0.3 --pref_rank_margin 0.1 --pref_rank_weight 1.0 \
--pref_loss_type margin \
--use_intervention --intervention_mode agent --teacher_type ${TEACHER_TYPE} \
--intervention_episode_prob 1.0 --intervention_episode_prob_min 1.0 \
--intervention_episode_prob_decay_steps 0 --intervention_episode_prob_decay_start 0 \
--reward_type ${REWARD_TYPE}"

launch_tol_run () {
  local GPU_ID=$1
  local TOL=$2
  local TOL_TAG
  TOL_TAG=$(echo "${TOL}" | tr '.' 'p')

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    ${BASE_ARGS} \
    --tolerance_type "${TOLERANCE_TYPE}" --tolerance_value "${TOL}" \
    --exp_name "own_${TEACHER_TYPE}_${OGBENCH_ENV}_tol_${TOL_TAG}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/own_${TEACHER_TYPE}_${OGBENCH_ENV}_tol_${TOL_TAG}_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
}

echo "Launching short cube tolerance sweep on ${OGBENCH_ENV}"

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
  echo "One or more cube tolerance sweep runs failed."
  exit 1
fi

echo "Cube tolerance sweep completed successfully."
