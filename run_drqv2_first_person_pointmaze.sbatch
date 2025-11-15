#!/bin/bash
#SBATCH --job-name=drqv2_pointmaze_fp
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pointmaze_fp_%j.out
#SBATCH --error=logs/drqv2_pointmaze_fp_%j.err

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
PROJECT=${PROJECT:-ogbench_drqv2}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-0}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}
FP_DISTANCE=${FP_DISTANCE:-3.0}
FP_HEIGHT=${FP_HEIGHT:-1.0}
FP_LOOKAHEAD=${FP_LOOKAHEAD:-2.0}
FP_PITCH=${FP_PITCH:--15.0}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
--viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE}"

BASE_ARGS="${BASE_ARGS} --pixel_camera_mode first_person \
--pixel_first_person_distance ${FP_DISTANCE} \
--pixel_first_person_height ${FP_HEIGHT} \
--pixel_first_person_lookahead ${FP_LOOKAHEAD} \
--pixel_first_person_pitch ${FP_PITCH}"

echo "Launching DrQ-v2 first-person arena (${ENV_A}) and danger (${ENV_B}) runs"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_A} \
  --exp_name "drqv2_${ENV_A}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${ENV_A}_drqv2_${SLURM_JOB_ID}.log &
PID_A=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_B} \
  --exp_name "drqv2_${ENV_B}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${ENV_B}_drqv2_${SLURM_JOB_ID}.log &
PID_B=$!

echo "PIDs: $PID_A (arena)  $PID_B (danger)"

wait $PID_A; EXIT_A=$?
wait $PID_B; EXIT_B=$?

if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ]; then
  echo "One or more DrQ-v2 runs failed ($EXIT_A, $EXIT_B)"
  exit 1
fi

echo "DrQ-v2 first-person PointMaze runs completed successfully."
