#!/bin/bash
#SBATCH --job-name=drqv2_compare_own_pvp
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_compare_own_pvp_%j.out
#SBATCH --error=logs/drqv2_compare_own_pvp_%j.err

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
PROJECT=${PROJECT:-ogbench_compare_own_vs_pvp}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TOLERANCE=${TOLERANCE:-30}
TEACHER_TYPE=${TEACHER_TYPE:-bfs}
INTERVENTION_MODE=${INTERVENTION_MODE:-agent_safety_align}
INTERVENTION_MARGIN=${INTERVENTION_MARGIN:-0.25}
INTERVENTION_RELEASE=${INTERVENTION_RELEASE:-3}
PREF_WEIGHT=${PREF_WEIGHT:-1.0}
PREF_MARGIN=${PREF_MARGIN:-0.1}
PREF_RATIO=${PREF_RATIO:-0.3}
DENSE_SCALE=${DENSE_SCALE:-0.05}
PVP_REWARD_TYPE=${PVP_REWARD_TYPE:-sparse}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint \
--use_intervention --intervention_mode ${INTERVENTION_MODE} --teacher_type ${TEACHER_TYPE} \
--tolerance_type angle --tolerance_value ${TOLERANCE} \
--intervention_safety_margin_frac ${INTERVENTION_MARGIN} \
--intervention_release_steps ${INTERVENTION_RELEASE}"

OWN_ARGS="--pref_buffer_enable --pref_capacity 200000 --pref_chunk_size 128 \
--pref_sample_ratio ${PREF_RATIO} --pref_rank_margin ${PREF_MARGIN} \
--pref_rank_weight ${PREF_WEIGHT} --pref_loss_type margin"

launch_own_run () {
  local GPU_ID=$1
  local REWARD_MODE=$2
  local EXTRA_ARGS=$3

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    --reward_type ${REWARD_MODE} \
    ${EXTRA_ARGS} \
    ${BASE_ARGS} \
    ${OWN_ARGS} \
    --exp_name "own_${TEACHER_TYPE}_${OGBENCH_ENV}_${REWARD_MODE}_margin_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/own_${TEACHER_TYPE}_${OGBENCH_ENV}_${REWARD_MODE}_margin_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started own ${REWARD_MODE} on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
  PIDS+=(${LAUNCHED_PID})
}

launch_pvp_run () {
  local GPU_ID=$1

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_pvp_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    --reward_type ${PVP_REWARD_TYPE} \
    ${BASE_ARGS} \
    --exp_name "pvp_${TEACHER_TYPE}_${OGBENCH_ENV}_noreward_pvp_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/pvp_${TEACHER_TYPE}_${OGBENCH_ENV}_noreward_pvp_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started PVP noreward on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
  PIDS+=(${LAUNCHED_PID})
}

echo "Launching comparison runs (own: dense/sparse/combined + PVP noreward)"

PIDS=()
launch_own_run 0 dense "--dense_reward_scale ${DENSE_SCALE}"
launch_own_run 1 sparse "--dense_reward_scale 0.0"
launch_own_run 2 combined "--dense_reward_scale ${DENSE_SCALE}"
launch_pvp_run 3

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more comparison runs failed."
  exit 1
fi

echo "DrQ-v2 comparison sweep completed successfully."
