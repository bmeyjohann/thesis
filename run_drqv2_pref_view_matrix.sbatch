#!/bin/bash
#SBATCH --job-name=drqv2_pref_views
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pref_views_%j.out
#SBATCH --error=logs/drqv2_pref_views_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

SAFE_ENV=${SAFE_ENV:-visual-pointmaze-arena-v0}
DANGER_ENV=${DANGER_ENV:-visual-pointmaze-arena-danger-lethal-v0}
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
PREF_LOSS=${PREF_LOSS:-margin}
FP_DISTANCE=${FP_DISTANCE:-3.0}
FP_HEIGHT=${FP_HEIGHT:-1.0}
FP_LOOKAHEAD=${FP_LOOKAHEAD:-2.0}
FP_PITCH=${FP_PITCH:--15.0}
LOCAL_VIEW_SIZE=${LOCAL_VIEW_SIZE:-25.0}
LOCAL_CAMERA_HEIGHT=${LOCAL_CAMERA_HEIGHT:-""}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint --pref_buffer_enable \
--pref_capacity 200000 --pref_chunk_size 128 --pref_sample_ratio ${PREF_RATIO} \
--pref_rank_margin ${PREF_MARGIN} --pref_rank_weight ${PREF_WEIGHT} \
--pref_loss_type ${PREF_LOSS} --use_intervention --intervention_mode agent \
--teacher_type bfs --tolerance_type angle --tolerance_value ${TOLERANCE}"

launch_run () {
  local GPU_ID=$1
  local ENV_NAME=$2
  local TAG=$3
  local VIEW_MODE=$4

  local RUN_ARGS="${BASE_ARGS}"
  if [ "${VIEW_MODE}" = "first_person" ]; then
    RUN_ARGS="${RUN_ARGS} --pixel_camera_mode first_person \
--pixel_first_person_distance ${FP_DISTANCE} \
--pixel_first_person_height ${FP_HEIGHT} \
--pixel_first_person_lookahead ${FP_LOOKAHEAD} \
--pixel_first_person_pitch ${FP_PITCH}"
  else
    RUN_ARGS="${RUN_ARGS} --pixel_camera_mode agent_local \
--pixel_local_view_size ${LOCAL_VIEW_SIZE}"
    if [ -n "${LOCAL_CAMERA_HEIGHT}" ]; then
      RUN_ARGS="${RUN_ARGS} --pixel_local_camera_height ${LOCAL_CAMERA_HEIGHT}"
    fi
  fi

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${ENV_NAME}" \
    --exp_name "drqv2_${TAG}_${VIEW_MODE}_${SLURM_JOB_ID}" \
    ${RUN_ARGS} \
    2>&1 | tee "logs/${TAG}_${VIEW_MODE}_drqv2_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${TAG}/${VIEW_MODE} on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
}

echo "Launching DrQ-v2 preference runs (view mode x env matrix)"

PIDS=()
launch_run 0 "${SAFE_ENV}" safe first_person
PIDS+=(${LAUNCHED_PID})
launch_run 1 "${DANGER_ENV}" danger first_person
PIDS+=(${LAUNCHED_PID})
launch_run 2 "${SAFE_ENV}" safe agent_local
PIDS+=(${LAUNCHED_PID})
launch_run 3 "${DANGER_ENV}" danger agent_local
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
  echo "One or more preference view runs failed (codes: ${EXIT_CODES[*]})"
  exit 1
fi

echo "DrQ-v2 preference view matrix completed successfully."
