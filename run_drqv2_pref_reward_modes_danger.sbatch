#!/bin/bash
#SBATCH --job-name=drqv2_pref_reward
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pref_reward_%j.out
#SBATCH --error=logs/drqv2_pref_reward_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

OGBENCH_ENV=${OGBENCH_ENV:-pointmaze-arena-danger-lethal-v0}
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

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint --pref_buffer_enable \
--pref_capacity 200000 --pref_chunk_size 128 --pref_sample_ratio ${PREF_RATIO} \
--pref_rank_margin ${PREF_MARGIN} --pref_rank_weight ${PREF_WEIGHT} \
--use_intervention --intervention_mode agent --teacher_type bfs \
--tolerance_type angle --tolerance_value ${TOLERANCE}"

launch_reward_run () {
  local GPU_ID=$1
  local REWARD_MODE=$2
  local EXTRA_ARGS=$3

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    --reward_type ${REWARD_MODE} \
    ${EXTRA_ARGS} \
    ${BASE_ARGS} \
    --exp_name "drqv2_${OGBENCH_ENV}_${REWARD_MODE}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/${OGBENCH_ENV}_${REWARD_MODE}_drqv2_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${REWARD_MODE} reward run on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
}

echo "Launching DrQ-v2 preference runs for sparse vs dense reward"

PIDS=()
launch_reward_run 0 dense "--dense_reward_scale 0.05"
PIDS+=(${LAUNCHED_PID})
launch_reward_run 1 sparse "--dense_reward_scale 0.0"
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
  echo "One or more reward-mode runs failed (codes: ${EXIT_CODES[*]})"
  exit 1
fi

echo "DrQ-v2 reward-mode comparison completed successfully."
