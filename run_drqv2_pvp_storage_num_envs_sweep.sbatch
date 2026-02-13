#!/bin/bash
# Sweep num_envs for maze PVP storage runs (one run per GPU).
# Default storage mode is tmp; set STORAGE_MODE=scratch or STORAGE_MODE=ram to compare.
#
# Example:
#   bash submit_job.sh run_drqv2_pvp_storage_num_envs_sweep.sbatch
#   STORAGE_MODE=ram NUM_ENVS_SWEEP="16 32 48 64" bash submit_job.sh run_drqv2_pvp_storage_num_envs_sweep.sbatch

#SBATCH --job-name=drqv2_pvp_nenv_sweep
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/drqv2_pvp_nenv_sweep_%j.out
#SBATCH --error=logs/drqv2_pvp_nenv_sweep_%j.err

set -euo pipefail

mkdir -p logs models

cd sc_venv_template
source activate.sh
cd ..

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SLURM_TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID}}
mkdir -p "${XDG_CACHE_HOME}"
export WANDB_DIR=${WANDB_DIR:-$PWD}
mkdir -p "${WANDB_DIR}/wandb"

TMP_BUFFER_ROOT=${TMP_BUFFER_ROOT:-/tmp/drqv2_buffers_${SLURM_JOB_ID}}
SCRATCH_BUFFER_ROOT=${SCRATCH_BUFFER_ROOT:-${CSCRATCH:-${SCRATCH:-/tmp}}/drqv2_buffers_${SLURM_JOB_ID}}
mkdir -p "${TMP_BUFFER_ROOT}" "${SCRATCH_BUFFER_ROOT}"
echo "Using TMP_BUFFER_ROOT=${TMP_BUFFER_ROOT}"
echo "Using SCRATCH_BUFFER_ROOT=${SCRATCH_BUFFER_ROOT}"

CLEANUP_BUFFERS=${CLEANUP_BUFFERS:-1}
cleanup_buffers() {
  if [ "${CLEANUP_BUFFERS}" = "1" ]; then
    if [ -n "${TMP_BUFFER_ROOT:-}" ] && [ -d "${TMP_BUFFER_ROOT}" ]; then
      rm -rf "${TMP_BUFFER_ROOT}"
      echo "Cleaned TMP_BUFFER_ROOT=${TMP_BUFFER_ROOT}"
    fi
    if [ -n "${SCRATCH_BUFFER_ROOT:-}" ] && [ -d "${SCRATCH_BUFFER_ROOT}" ] && [ "${SCRATCH_BUFFER_ROOT}" != "${TMP_BUFFER_ROOT}" ]; then
      rm -rf "${SCRATCH_BUFFER_ROOT}"
      echo "Cleaned SCRATCH_BUFFER_ROOT=${SCRATCH_BUFFER_ROOT}"
    fi
  fi
}
trap cleanup_buffers EXIT

OGBENCH_ENV=${OGBENCH_ENV:-visual-pointmaze-arena-danger-lethal-v0}
NUM_ENVS_SWEEP=${NUM_ENVS_SWEEP:-"16 32 48 64"}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
NUM_EXPL_STEPS=${NUM_EXPL_STEPS:-20000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LOG_INTERVAL=${LOG_INTERVAL:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
EVAL_EVERY=${EVAL_EVERY:-50000}
PROJECT=${PROJECT:-ogbench_drqv2_pvp_storage_num_envs_sweep}
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
STORAGE_MODE=${STORAGE_MODE:-tmp}  # tmp|scratch|ram

BASE_ARGS="--obs_mode pixels --pixel_width 84 --pixel_height 84 \
--total_timesteps ${TOTAL_STEPS} --learning_rate ${LEARNING_RATE} \
--num_expl_steps ${NUM_EXPL_STEPS} --save_interval ${SAVE_INTERVAL} \
--eval_every_frames ${EVAL_EVERY} --project ${PROJECT} \
--frame_stack ${FRAME_STACK} --action_repeat ${ACTION_REPEAT} \
--drq_feature_dim ${FEATURE_DIM} --drq_hidden_dim ${HIDDEN_DIM} \
--use_wandb --log_interval ${LOG_INTERVAL} \
--use_intervention --teacher_type ${TEACHER_TYPE} \
--reward_type ${PVP_REWARD_TYPE} \
--replay_buffer_num_workers 0 \
--intervention_mode agent --tolerance_type angle --tolerance_value ${TOLERANCE} \
--nstep 1"

launch_run () {
  local GPU_ID=$1
  local NENV=$2
  local EXTRA_ARGS
  local RUN_BUFFER_ROOT

  case "${STORAGE_MODE}" in
    tmp)
      RUN_BUFFER_ROOT="${TMP_BUFFER_ROOT}/nenv_${NENV}_tmp"
      EXTRA_ARGS="--pref_chunk_size ${PREF_CHUNK_SIZE} --pref_fetch_every ${PREF_FETCH_EVERY}"
      ;;
    scratch)
      RUN_BUFFER_ROOT="${SCRATCH_BUFFER_ROOT}/nenv_${NENV}_scratch"
      EXTRA_ARGS="--pref_chunk_size ${PREF_CHUNK_SIZE} --pref_fetch_every ${PREF_FETCH_EVERY}"
      ;;
    ram)
      RUN_BUFFER_ROOT="${SCRATCH_BUFFER_ROOT}/nenv_${NENV}_ram"
      EXTRA_ARGS="--replay_mode ram --replay_buffer_size ${RAM_REPLAY_SIZE}"
      ;;
    *)
      echo "Unknown STORAGE_MODE=${STORAGE_MODE} (expected tmp|scratch|ram)"
      exit 2
      ;;
  esac

  mkdir -p "${RUN_BUFFER_ROOT}"
  WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train_drqv2_pvp_ogbench.py \
    --env_name "${OGBENCH_ENV}" \
    --num_envs "${NENV}" \
    ${BASE_ARGS} \
    --buffer_root "${RUN_BUFFER_ROOT}" \
    ${EXTRA_ARGS} \
    --exp_name "pvp_${STORAGE_MODE}_${OGBENCH_ENV}_nenv_${NENV}_${SLURM_JOB_ID}" \
    2>&1 | tee "logs/pvp_${STORAGE_MODE}_${OGBENCH_ENV}_nenv_${NENV}_${SLURM_JOB_ID}.log" &
  PIDS+=($!)
  echo "Started nenv=${NENV} storage=${STORAGE_MODE} on GPU ${GPU_ID}"
}

echo "Launching PVP num_envs sweep on ${OGBENCH_ENV} with STORAGE_MODE=${STORAGE_MODE}"
echo "NUM_ENVS_SWEEP=${NUM_ENVS_SWEEP}"

GPU_IDS=(0 1 2 3)
IDX=0
PIDS=()
for NENV in ${NUM_ENVS_SWEEP}; do
  GPU_ID=${GPU_IDS[$IDX]}
  launch_run "${GPU_ID}" "${NENV}"
  IDX=$(( (IDX + 1) % ${#GPU_IDS[@]} ))
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

if [ "${FAIL}" -ne 0 ]; then
  echo "One or more num_envs sweep runs failed."
  exit 1
fi

echo "PVP num_envs sweep completed successfully."
