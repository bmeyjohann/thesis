#!/usr/bin/env bash
set -euo pipefail

# Runs 3 dense-reward manipulation experiments on a fixed-reset cube setup:
# 1) Dense reward only (no teacher intervention)
# 2) Dense reward + online teacher intervention
# 3) Dense reward + demo prefill only (no online intervention)
#
# Optional overrides:
#   PROJECT=ogbench-manip-reward-debug TOTAL_TIMESTEPS=120000 STATIC_RESET_SEED=123 \
#   bash scripts/run_cube_dense_triplet_local.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_ENVS="${NUM_ENVS:-8}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
STATIC_RESET_SEED="${STATIC_RESET_SEED:-123}"

TEACHER_TYPE="${TEACHER_TYPE:-cube_plan}"
TOL_TYPE="${TOL_TYPE:-l2}"
TOL_VALUE="${TOL_VALUE:-0.05}"

DEMO_PREFILL_STEPS="${DEMO_PREFILL_STEPS:-10000}"
DEMO_BUFFER_CAPACITY="${DEMO_BUFFER_CAPACITY:-100000}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
PROFILE_TIMING="${PROFILE_TIMING:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --obs_mode state
  --num_envs "${NUM_ENVS}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --reward_type sparse
  --cube_reward_mode dense
  --cube_dense_progress_scale 1.0
  --static_reset_seed "${STATIC_RESET_SEED}"
  --teacher_type "${TEACHER_TYPE}"
  --tolerance_type "${TOL_TYPE}"
  --tolerance_value "${TOL_VALUE}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --save_interval "${SAVE_INTERVAL}"
  --use_wandb
  --project "${PROJECT}"
)
if [[ "${PROFILE_TIMING}" != "0" ]]; then
  COMMON_ARGS+=(--profile_timing)
fi

run_one() {
  local exp_name="$1"
  shift
  local log_file="logs/${exp_name}.log"
  echo "=== Starting ${exp_name} ==="
  WANDB_MODE=online \
  WANDB_CONSOLE=off \
  WANDB_SILENT=true \
  "${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
    "${COMMON_ARGS[@]}" \
    --exp_name "${exp_name}" \
    "$@" \
    2>&1 | tee "${log_file}" &
  local pid=$!
  echo "=== Launched ${exp_name} (pid=${pid}) ==="
  RUN_PIDS+=("${pid}")
  RUN_NAMES+=("${exp_name}")
}

RUN_PIDS=()
RUN_NAMES=()

run_one "cube_dense_no_intervention_${TS}"
run_one "cube_dense_with_intervention_${TS}" \
  --use_intervention \
  --intervention_mode agent
run_one "cube_dense_demo_prefill_only_${TS}" \
  --demo_buffer_enable \
  --demo_buffer_capacity "${DEMO_BUFFER_CAPACITY}" \
  --demo_prefill_steps "${DEMO_PREFILL_STEPS}" \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent \
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"

echo "=== Waiting for concurrent runs to finish ==="
for i in "${!RUN_PIDS[@]}"; do
  pid="${RUN_PIDS[$i]}"
  name="${RUN_NAMES[$i]}"
  if wait "${pid}"; then
    echo "=== Finished ${name} (pid=${pid}) ==="
  else
    echo "=== FAILED ${name} (pid=${pid}) ==="
    exit 1
  fi
done

echo "=== All 3 runs completed successfully ==="
