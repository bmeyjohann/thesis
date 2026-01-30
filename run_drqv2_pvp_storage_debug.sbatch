#!/bin/bash
#SBATCH --job-name=drqv2_pvp_storage_debug
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pvp_storage_debug_%j.out
#SBATCH --error=logs/drqv2_pvp_storage_debug_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

OGBENCH_ENV=${OGBENCH_ENV:-visual-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
EVAL_EVERY=${EVAL_EVERY:-50000}
PROJECT=${PROJECT:-ogbench_pvp_storage_debug}
FRAME_STACK=${FRAME_STACK:-3}
ACTION_REPEAT=${ACTION_REPEAT:-2}
FEATURE_DIM=${FEATURE_DIM:-50}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
TOLERANCE=${TOLERANCE:-30}
TEACHER_TYPE=${TEACHER_TYPE:-bfs}
PVP_REWARD_TYPE=${PVP_REWARD_TYPE:-sparse}
PREF_CHUNK_SIZE=${PREF_CHUNK_SIZE:-64}
PREF_FETCH_EVERY=${PREF_FETCH_EVERY:-512}
RAM_REPLAY_SIZE=${RAM_REPLAY_SIZE:-500000}

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --viz_on_checkpoint \
--use_intervention --teacher_type ${TEACHER_TYPE} \
--reward_type ${PVP_REWARD_TYPE} \
--replay_buffer_num_workers 0 \
--intervention_mode agent --tolerance_type angle --tolerance_value ${TOLERANCE}"

launch_pvp_run () {
  local GPU_ID=$1
  local MODE=$2
  local EXTRA_ARGS=$3

  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_pvp_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    ${BASE_ARGS} \
    ${EXTRA_ARGS} \
    --exp_name "pvp_storage_${MODE}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/pvp_storage_${MODE}_${SLURM_JOB_ID}.log" &
  LAUNCHED_PID=$!
  echo "Started ${MODE} on GPU ${GPU_ID} (PID ${LAUNCHED_PID})"
  PIDS+=(${LAUNCHED_PID})
}

echo "Launching PVP storage debug variants"

PIDS=()
launch_pvp_run 0 "baseline" \
  "--pref_chunk_size ${PREF_CHUNK_SIZE} --pref_fetch_every ${PREF_FETCH_EVERY}"
launch_pvp_run 1 "chunkfetch" \
  "--pref_chunk_size 512 --pref_fetch_every 5000"
launch_pvp_run 2 "tmpdir" \
  "--buffer_root ${TMPDIR:-/tmp}/pvp_buffers_${SLURM_JOB_ID}_tmp"
launch_pvp_run 3 "ramdisk" \
  "--buffer_root /dev/shm/pvp_buffers_${SLURM_JOB_ID}_ram --replay_buffer_size ${RAM_REPLAY_SIZE}"

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more PVP storage debug runs failed."
  exit 1
fi

echo "PVP storage debug sweep completed successfully."
