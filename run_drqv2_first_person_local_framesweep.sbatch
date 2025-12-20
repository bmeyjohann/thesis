#!/bin/bash
#SBATCH --job-name=drqv2_fp_fsweep
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_fp_fsweep_%j.out
#SBATCH --error=logs/drqv2_fp_fsweep_%j.err

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
FRAME_STACK_A=${FRAME_STACK_A:-10}
FRAME_STACK_B=${FRAME_STACK_B:-20}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-0}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}
SE2_SCALE=${SE2_SCALE:-0.25}
RUN_LABEL=${RUN_LABEL:-no_teacher}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--action_repeat ${ACTION_REPEAT} --drq_feature_dim ${FEATURE_DIM} \
--drq_hidden_dim ${HIDDEN_DIM} --use_wandb --viz_on_checkpoint \
--viz_first_step ${VIZ_FIRST_STEP} --viz_grid_resolution ${VIZ_GRID_RES} \
--viz_quiver_stride ${VIZ_QUIVER_STRIDE} --use_local_actions \
--se2_translation_scale ${SE2_SCALE} --goal_marker_color green"

echo "Launching DrQ-v2 first-person local frame stack sweep (no teacher)"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_A} \
  --frame_stack ${FRAME_STACK_A} \
  --exp_name "drqv2_fp_local_${RUN_LABEL}_${ENV_A}_fs${FRAME_STACK_A}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${RUN_LABEL}_${ENV_A}_fs${FRAME_STACK_A}_drqv2_${SLURM_JOB_ID}.log &
PID_A=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_A} \
  --frame_stack ${FRAME_STACK_B} \
  --exp_name "drqv2_fp_local_${RUN_LABEL}_${ENV_A}_fs${FRAME_STACK_B}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${RUN_LABEL}_${ENV_A}_fs${FRAME_STACK_B}_drqv2_${SLURM_JOB_ID}.log &
PID_B=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_B} \
  --frame_stack ${FRAME_STACK_A} \
  --exp_name "drqv2_fp_local_${RUN_LABEL}_${ENV_B}_fs${FRAME_STACK_A}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${RUN_LABEL}_${ENV_B}_fs${FRAME_STACK_A}_drqv2_${SLURM_JOB_ID}.log &
PID_C=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_B} \
  --frame_stack ${FRAME_STACK_B} \
  --exp_name "drqv2_fp_local_${RUN_LABEL}_${ENV_B}_fs${FRAME_STACK_B}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${RUN_LABEL}_${ENV_B}_fs${FRAME_STACK_B}_drqv2_${SLURM_JOB_ID}.log &
PID_D=$!

echo "PIDs: $PID_A $PID_B $PID_C $PID_D"

wait $PID_A; EA=$?
wait $PID_B; EB=$?
wait $PID_C; EC=$?
wait $PID_D; ED=$?

if [ $EA -ne 0 ] || [ $EB -ne 0 ] || [ $EC -ne 0 ] || [ $ED -ne 0 ]; then
  echo "One or more DrQ-v2 frame stack sweep runs failed: $EA $EB $EC $ED"
  exit 1
fi

echo "DrQ-v2 first-person local frame stack sweep (no teacher) completed."
