#!/bin/bash
#SBATCH --job-name=drqv2_fp_recurrent_se2
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_fp_recurrent_se2_%j.out
#SBATCH --error=logs/drqv2_fp_recurrent_se2_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

TARGET_ENV=${TARGET_ENV:-visual-pointmaze-arena-v0}
TOTAL_STEPS=${TOTAL_STEPS:-800000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
EVAL_EVERY=${EVAL_EVERY:-100000}
PROJECT=${PROJECT:-ogbench_drqv2}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
GRU_HIDDEN=${GRU_HIDDEN:-512}
CONV_CHANNELS=${CONV_CHANNELS:-64}
UNROLL=${UNROLL:-8}
BURN_IN=${BURN_IN:-4}
FP_DISTANCE=${FP_DISTANCE:-3.0}
FP_HEIGHT=${FP_HEIGHT:-1.0}
FP_LOOKAHEAD=${FP_LOOKAHEAD:-2.0}
FP_PITCH=${FP_PITCH:--15.0}
VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-0}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--agent_variant recurrent --recurrent_type convgru \
--recurrent_conv_channels ${CONV_CHANNELS} --recurrent_hidden_dim ${GRU_HIDDEN} \
--recurrent_unroll_length ${UNROLL} --recurrent_burn_in ${BURN_IN} \
--use_local_actions --use_wandb \
--viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
--viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE} \
--pixel_camera_mode first_person \
--pixel_first_person_distance ${FP_DISTANCE} \
--pixel_first_person_height ${FP_HEIGHT} \
--pixel_first_person_lookahead ${FP_LOOKAHEAD} \
--pixel_first_person_pitch ${FP_PITCH} \
--goal_marker_color green"

launch_run () {
  local GPU_ID=$1
  local SE2_SCALE=$2
  local TAG=$3

  local RUN_ARGS="${BASE_ARGS}"
  if (( $(echo "${SE2_SCALE} > 0" | bc -l) )); then
    RUN_ARGS="${RUN_ARGS} --recurrent_use_se2_warp --se2_translation_scale ${SE2_SCALE}"
  fi

  local EXP_NAME="drqv2_se2_${TAG}_${SLURM_JOB_ID}"
  local LOG_PATH="logs/drqv2_se2_${TAG}_${SLURM_JOB_ID}.log"

  echo "Launching ${EXP_NAME} on GPU ${GPU_ID} (se2_scale=${SE2_SCALE})"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${TARGET_ENV}" \
    --exp_name "${EXP_NAME}" \
    ${RUN_ARGS} \
    2>&1 | tee "${LOG_PATH}" &
  PIDS+=($!)
}

PIDS=()

launch_run 0 0.0 "no_se2"
launch_run 1 0.02 "se2_0p02"
launch_run 2 0.1 "se2_0p1"
launch_run 3 0.5 "se2_0p5"

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "❌ One or more SE2 sweep runs failed."
  exit 1
fi

echo "✅ Completed SE2 recurrent sweep (none, 0.02, 0.1, 0.5)."
