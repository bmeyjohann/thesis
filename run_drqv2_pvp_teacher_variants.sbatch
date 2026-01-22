#!/bin/bash
#SBATCH --job-name=drqv2_pvp_teacher_variants
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pvp_teacher_variants_%j.out
#SBATCH --error=logs/drqv2_pvp_teacher_variants_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

OGBENCH_ENV=${OGBENCH_ENV:-visual-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-1200000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
EVAL_EVERY=${EVAL_EVERY:-50000}
PROJECT=${PROJECT:-ogbench_pvp_teacher_variants}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TOLERANCE=${TOLERANCE:-30}
TEACHER_TYPE=${TEACHER_TYPE:-bfs}
INTERVENTION_MARGIN=${INTERVENTION_MARGIN:-0.25}
PVP_REWARD_TYPE=${PVP_REWARD_TYPE:-sparse}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint \
--use_intervention --teacher_type ${TEACHER_TYPE} \
--reward_type ${PVP_REWARD_TYPE}"

launch_pvp_run () {
  local GPU_ID=$1
  local MODE=$2
  local EXTRA_ARGS=$3

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_pvp_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    ${BASE_ARGS} \
    ${EXTRA_ARGS} \
    --exp_name "pvp_${TEACHER_TYPE}_${OGBENCH_ENV}_${MODE}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/pvp_${TEACHER_TYPE}_${OGBENCH_ENV}_${MODE}_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${MODE} on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
  PIDS+=(${LAUNCHED_PID})
}

echo "Launching PVP teacher variants"

PIDS=()
launch_pvp_run 0 "safety_align_release3" \
  "--intervention_mode agent_safety_align --intervention_safety_margin_frac ${INTERVENTION_MARGIN} --intervention_release_steps 3"
launch_pvp_run 1 "safety_progress_enable3_release3" \
  "--intervention_mode agent_safety_progress --intervention_enable_after_steps 3 --intervention_safety_margin_frac ${INTERVENTION_MARGIN} --intervention_release_steps 3"
launch_pvp_run 2 "safety_progress_enable10_release10" \
  "--intervention_mode agent_safety_progress --intervention_enable_after_steps 10 --intervention_safety_margin_frac ${INTERVENTION_MARGIN} --intervention_release_steps 10"
launch_pvp_run 3 "tolerance_angle30" \
  "--intervention_mode agent --tolerance_type angle --tolerance_value ${TOLERANCE}"

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more PVP runs failed."
  exit 1
fi

echo "PVP teacher variants completed successfully."
