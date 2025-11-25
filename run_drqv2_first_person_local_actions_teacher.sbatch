#!/bin/bash
#SBATCH --job-name=drqv2_fp_local_teacher
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_fp_local_teacher_%j.out
#SBATCH --error=logs/drqv2_fp_local_teacher_%j.err

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
SE2_SCALE=${SE2_SCALE:-0.25}
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
--use_local_actions --se2_translation_scale ${SE2_SCALE} \
--use_intervention --intervention_mode agent --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOLERANCE} \
--goal_marker_color green"

echo "Launching DrQ-v2 first-person local runs with teacher interventions"

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_A} \
  --exp_name "drqv2_${RUN_LABEL}_${ENV_A}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${RUN_LABEL}_${ENV_A}_drqv2_${SLURM_JOB_ID}.log &
PID_A=$!

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -u train_drqv2_ogbench.py \
  --env_name ${ENV_B} \
  --exp_name "drqv2_${RUN_LABEL}_${ENV_B}_$SLURM_JOB_ID" \
  ${BASE_ARGS} \
  2>&1 | tee logs/${RUN_LABEL}_${ENV_B}_drqv2_${SLURM_JOB_ID}.log &
PID_B=$!

echo "PIDs: $PID_A (arena teacher)  $PID_B (danger teacher)"

wait $PID_A; EXIT_A=$?
wait $PID_B; EXIT_B=$?

if [ $EXIT_A -ne 0 ] || [ $EXIT_B -ne 0 ]; then
  echo "One or more DrQ-v2 local teacher runs failed ($EXIT_A, $EXIT_B)"
  exit 1
fi

echo "DrQ-v2 first-person local teacher runs completed successfully."
