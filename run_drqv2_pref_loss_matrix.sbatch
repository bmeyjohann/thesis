#!/bin/bash
#SBATCH --job-name=drqv2_pref_loss
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pref_loss_%j.out
#SBATCH --error=logs/drqv2_pref_loss_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

SAFE_ENV=${SAFE_ENV:-pointmaze-arena-v0}
DANGER_ENV=${DANGER_ENV:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-1200000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
EVAL_EVERY=${EVAL_EVERY:-200000}
PROJECT=${PROJECT:-ogbench_drqv2_pref}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TOLERANCE=${TOLERANCE:-30}
PREF_WEIGHT=${PREF_WEIGHT:-1.0}
PREF_MARGIN=${PREF_MARGIN:-0.1}
PREF_RATIO=${PREF_RATIO:-0.3}
VIZ_GRID_RES=${VIZ_GRID_RES:-48}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint --viz_grid_resolution ${VIZ_GRID_RES} \
--viz_quiver_stride ${VIZ_QUIVER_STRIDE} --pref_buffer_enable \
--pref_capacity 200000 --pref_chunk_size 128 --pref_sample_ratio ${PREF_RATIO} \
--pref_rank_margin ${PREF_MARGIN} --pref_rank_weight ${PREF_WEIGHT} \
--use_intervention --intervention_mode agent --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOLERANCE}"

launch_pref_run () {
  local GPU_ID=$1
  local ENV_NAME=$2
  local LOSS_TYPE=$3
  local TAG=$4

  local RUN_ARGS="${BASE_ARGS} --pref_loss_type ${LOSS_TYPE}"

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${ENV_NAME}" \
    --exp_name "drqv2_${TAG}_${LOSS_TYPE}_${SLURM_JOB_ID}" \
    ${RUN_ARGS} \
    2>&1 | tee "logs/${TAG}_${LOSS_TYPE}_drqv2_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${TAG}/${LOSS_TYPE} on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
}

echo "Launching DrQ-v2 preference-loss matrix (env x loss)"

PIDS=()
launch_pref_run 0 "${SAFE_ENV}" margin safe
PIDS+=(${LAUNCHED_PID})
launch_pref_run 1 "${SAFE_ENV}" bradley_terry safe
PIDS+=(${LAUNCHED_PID})
launch_pref_run 2 "${DANGER_ENV}" margin danger
PIDS+=(${LAUNCHED_PID})
launch_pref_run 3 "${DANGER_ENV}" bradley_terry danger
PIDS+=(${LAUNCHED_PID})

EXIT_CODES=()
for PID in "${PIDS[@]}"; do
  wait "${PID}"
  EXIT_CODES+=($?)
done

FAIL=0
for CODE in "${EXIT_CODES[@]}"; do
  if [ "${CODE}" -ne 0 ]; then
    FAIL=1
  fi
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more preference-loss runs failed (codes: ${EXIT_CODES[*]})"
  exit 1
fi

echo "DrQ-v2 preference-loss matrix completed successfully."
