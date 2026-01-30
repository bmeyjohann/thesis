#!/bin/bash
#SBATCH --job-name=drqv2_hilserl_vs_own
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_hilserl_vs_own_%j.out
#SBATCH --error=logs/drqv2_hilserl_vs_own_%j.err

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
PROJECT=${PROJECT:-ogbench_hilserl_vs_own}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TOLERANCE=${TOLERANCE:-30}
TEACHER_TYPE=${TEACHER_TYPE:-bfs}
INTERVENTION_MODE=${INTERVENTION_MODE:-agent_safety_align}
INTERVENTION_MARGIN=${INTERVENTION_MARGIN:-0.25}
INTERVENTION_RELEASE=${INTERVENTION_RELEASE:-3}
DEMO_PREFILL_EPISODES=${DEMO_PREFILL_EPISODES:-20}
REWARD_MODE=${REWARD_MODE:-sparse}

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
--intervention_release_steps ${INTERVENTION_RELEASE} \
--reward_type ${REWARD_MODE}"

OWN_PREF_ARGS="--pref_buffer_enable --pref_capacity 200000 --pref_chunk_size 128 \
--pref_sample_ratio 0.3 --pref_rank_margin 0.1 --pref_rank_weight 1.0 \
--pref_loss_type margin"

launch_hilserl () {
  local GPU_ID=$1
  local MODE=$2
  local EXTRA_ARGS=$3

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_hilserl_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    ${BASE_ARGS} \
    ${EXTRA_ARGS} \
    --exp_name "hilserl_${TEACHER_TYPE}_${OGBENCH_ENV}_${MODE}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/hilserl_${TEACHER_TYPE}_${OGBENCH_ENV}_${MODE}_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
}

launch_own () {
  local GPU_ID=$1
  local MODE=$2
  local EXTRA_ARGS=$3

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    ${BASE_ARGS} \
    ${OWN_PREF_ARGS} \
    ${EXTRA_ARGS} \
    --exp_name "own_${TEACHER_TYPE}_${OGBENCH_ENV}_${MODE}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/own_${TEACHER_TYPE}_${OGBENCH_ENV}_${MODE}_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
}

echo "Launching HIL-SERL vs own variants (sparse reward)"

PIDS=()
launch_hilserl 0 "demo50_prefill20" \
  "--demo_buffer_enable --demo_sample_ratio 0.5 --demo_prefill_episodes ${DEMO_PREFILL_EPISODES} --demo_prefill_target demo"

launch_own 1 "demo50_prefill20" \
  "--demo_buffer_enable --demo_sample_ratio 0.5 --demo_prefill_episodes ${DEMO_PREFILL_EPISODES} --demo_prefill_target demo"

launch_own 2 "replay_prefill20" \
  "--demo_sample_ratio 0.0 --demo_prefill_episodes ${DEMO_PREFILL_EPISODES} --demo_prefill_target replay"

launch_own 3 "no_prefill" \
  "--demo_sample_ratio 0.0 --demo_prefill_steps 0 --demo_prefill_episodes 0"

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more HIL-SERL vs own runs failed."
  exit 1
fi

echo "HIL-SERL vs own comparison completed successfully."
