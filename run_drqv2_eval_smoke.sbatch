#!/bin/bash
#SBATCH --job-name=drqv2_eval_smoke
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:3
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_eval_smoke_%j.out
#SBATCH --error=logs/drqv2_eval_smoke_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

ENV_B=${ENV_B:-visual-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-20000}
NUM_ENVS=${NUM_ENVS:-8}
EVAL_EVERY=${EVAL_EVERY:-5000}
NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES:-5}
PROJECT=${PROJECT:-ogbench_drqv2_eval_smoke}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-0}
VIZ_GRID_RES=${VIZ_GRID_RES:-32}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}
VIEW_SIZES=${VIEW_SIZES:-"25.0"}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-""}
TOLERANCE=${TOLERANCE:-30}
INTERVENTION_MODE=${INTERVENTION_MODE:-agent_safety_align}
INTERVENTION_MARGIN=${INTERVENTION_MARGIN:-0.25}
INTERVENTION_RELEASE=${INTERVENTION_RELEASE:-3}
REWARD_MODES=${REWARD_MODES:-"sparse dense none"}
RUN_LABEL=${RUN_LABEL:-eval_smoke_teacher}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate 1e-4 \
--num_expl_steps 2000 --save_interval 10000 \
--eval_every_frames ${EVAL_EVERY} --num_eval_episodes ${NUM_EVAL_EPISODES} \
--project ${PROJECT} --frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
--viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE} \
--use_intervention --intervention_mode ${INTERVENTION_MODE} --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOLERANCE} \
--intervention_safety_margin_frac ${INTERVENTION_MARGIN} \
--intervention_release_steps ${INTERVENTION_RELEASE} \
--goal_marker_color green"

launch_run () {
  local ENV_NAME=$1
  local VIEW_SIZE=$2
  local REWARD_MODE=$3
  local GPU_ID=$4

  local RUN_ARGS="${BASE_ARGS} --pixel_camera_mode agent_local --pixel_local_view_size ${VIEW_SIZE}"
  if [ -n "${CAMERA_HEIGHT}" ]; then
    RUN_ARGS="${RUN_ARGS} --pixel_local_camera_height ${CAMERA_HEIGHT}"
  fi

  RUN_ARGS="${RUN_ARGS} --reward_type ${REWARD_MODE}"

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${ENV_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --exp_name "drqv2_${RUN_LABEL}_${ENV_NAME}_v${VIEW_SIZE}_${REWARD_MODE}_${SLURM_JOB_ID}" \
    ${RUN_ARGS} \
    2>&1 | tee "logs/${RUN_LABEL}_${ENV_NAME}_v${VIEW_SIZE}_${REWARD_MODE}_drqv2_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${ENV_NAME}/${REWARD_MODE} on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
  PIDS+=(${LAUNCHED_PID})
}

echo "Launching DrQ-v2 eval smoke sweep (reward modes: ${REWARD_MODES})"

PIDS=()
GPU_IDS=(0 1 2)
GPU_INDEX=0

for REWARD_MODE in ${REWARD_MODES}; do
  for VIEW_SIZE in ${VIEW_SIZES}; do
    GPU_ID=${GPU_IDS[$GPU_INDEX]}
    launch_run "${ENV_B}" "${VIEW_SIZE}" "${REWARD_MODE}" "${GPU_ID}"
    GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPU_IDS[@]} ))
  done
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more DrQ-v2 eval smoke runs failed."
  exit 1
fi

echo "DrQ-v2 eval smoke sweep completed successfully."
