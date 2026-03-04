#!/usr/bin/env bash
set -euo pipefail

# Launch three cube-manip training runs in parallel, varying only cube_reward_mode.
#
# Defaults target "real" comparison runtime (not micro-smoke):
#   150k env steps, eval every 10k, save every 25k.
#
# Optional overrides:
#   ENV_NAME=cube-triple-singletask-task5-v0 \
#   NUM_ENVS=8 \
#   TOTAL_TIMESTEPS=150000 \
#   PROJECT=ogbench_cube_debug \
#   bash scripts/run_cube_reward_modes_online.sh

ENV_NAME="${ENV_NAME:-cube-triple-singletask-task5-v0}"
NUM_ENVS="${NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-150000}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
PROJECT="${PROJECT:-ogbench_cube_debug}"

# Student-with-interventions setup (constant across all 3 runs)
TOLERANCE_TYPE="${TOLERANCE_TYPE:-l2}"
TOLERANCE_VALUE="${TOLERANCE_VALUE:-0.05}"
PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_CAPACITY="${PREF_CAPACITY:-100000}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/cube_reward_modes_${TS}"
mkdir -p "${LOG_DIR}"

mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
NUM_GPUS="${#GPU_IDS[@]}"
RUN_IDX=0
PIDS=()

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --obs_mode state
  --num_envs "${NUM_ENVS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --save_interval "${SAVE_INTERVAL}"
  --reward_type sparse
  --use_intervention
  --intervention_mode agent
  --teacher_type cube_plan
  --tolerance_type "${TOLERANCE_TYPE}"
  --tolerance_value "${TOLERANCE_VALUE}"
  --pref_buffer_enable
  --pref_capacity "${PREF_CAPACITY}"
  --pref_sample_ratio "${PREF_SAMPLE_RATIO}"
  --pref_rank_weight "${PREF_RANK_WEIGHT}"
  --use_wandb
  --project "${PROJECT}"
)

cleanup() {
  if ((${#PIDS[@]} > 0)); then
    echo
    echo "Received interrupt. Stopping child runs: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

launch_run() {
  local name="$1"
  local mode="$2"
  local log_file="${LOG_DIR}/${name}.log"
  local gpu=""

  if (( NUM_GPUS > 0 )); then
    gpu="${GPU_IDS[$((RUN_IDX % NUM_GPUS))]}"
  fi

  echo "Launching ${name} (cube_reward_mode=${mode}, gpu=${gpu:-cpu}) -> ${log_file}"

  if [[ -n "${gpu}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
    WANDB_MODE=online \
    WANDB_CONSOLE=off \
    WANDB_SILENT=true \
    python train_fast_sac_ogbench_manip.py \
      "${COMMON_ARGS[@]}" \
      --cube_reward_mode "${mode}" \
      --exp_name "${name}" \
      > "${log_file}" 2>&1 &
  else
    WANDB_MODE=online \
    WANDB_CONSOLE=off \
    WANDB_SILENT=true \
    python train_fast_sac_ogbench_manip.py \
      "${COMMON_ARGS[@]}" \
      --cube_reward_mode "${mode}" \
      --exp_name "${name}" \
      > "${log_file}" 2>&1 &
  fi

  PIDS+=("$!")
  RUN_IDX=$((RUN_IDX + 1))
}

launch_run "cube_mode_sparse_final_${TS}" "sparse_final"
launch_run "cube_mode_sparse_intermediate_${TS}" "sparse_intermediate"
launch_run "cube_mode_dense_${TS}" "dense"

echo
echo "Started PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}"
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/*.log"
echo
echo "Wait for completion:"
echo "  wait ${PIDS[*]}"

wait "${PIDS[@]}"

echo
echo "All runs finished. Logs: ${LOG_DIR}"
