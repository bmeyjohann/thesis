#!/bin/bash
#SBATCH --job-name=drqv2_pointmaze_local_teacher
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pointmaze_local_teacher_%j.out
#SBATCH --error=logs/drqv2_pointmaze_local_teacher_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

ENV_A=${ENV_A:-visual-pointmaze-arena-v0}
ENV_B=${ENV_B:-visual-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-800000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
EVAL_EVERY=${EVAL_EVERY:-100000}
PROJECT=${PROJECT:-ogbench_drqv2_pref}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-0}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}
TOLERANCE=${TOLERANCE:-30}
RUN_LABEL=${RUN_LABEL:-teacher}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
--viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE} \
--use_intervention --intervention_mode agent --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOLERANCE} \
--goal_marker_color green"

launch_run () {
  local GPU_ID=$1
  local ENV_NAME=$2
  local EXP_SUFFIX=$3
  local VIEW_SIZE=$4
  local CAMERA_HEIGHT=$5

  local RUN_ARGS="${BASE_ARGS} --pixel_camera_mode agent_local --pixel_local_view_size ${VIEW_SIZE}"
  if [ -n "${CAMERA_HEIGHT}" ]; then
    RUN_ARGS="${RUN_ARGS} --pixel_local_camera_height ${CAMERA_HEIGHT}"
  fi

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${ENV_NAME}" \
    --exp_name "drqv2_${RUN_LABEL}_${ENV_NAME}_${EXP_SUFFIX}_${SLURM_JOB_ID}" \
    ${RUN_ARGS} \
    2>&1 | tee "logs/${RUN_LABEL}_${ENV_NAME}_${EXP_SUFFIX}_drqv2_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${ENV_NAME}/${EXP_SUFFIX} (${RUN_LABEL}) on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
}

echo "Launching DrQ-v2 local-view teacher runs (view sizes 25 and 50)"

PIDS=()
launch_run 0 "${ENV_A}" close 25.0 ""
PIDS+=(${LAUNCHED_PID})
launch_run 1 "${ENV_B}" close 25.0 ""
PIDS+=(${LAUNCHED_PID})
launch_run 2 "${ENV_A}" wide 50.0 ""
PIDS+=(${LAUNCHED_PID})
launch_run 3 "${ENV_B}" wide 50.0 ""
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
  echo "One or more DrQ-v2 local-view teacher runs failed (codes: ${EXIT_CODES[*]})"
  exit 1
fi

echo "DrQ-v2 local-view teacher PointMaze runs completed successfully."
