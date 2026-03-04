#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   NUM_ENVS=12 TOTAL_TIMESTEPS=120000 DEMO_PREFILL_EPISODES=20 bash scripts/run_cube_single_4way_online.sh
NUM_ENVS="${NUM_ENVS:-12}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
PROJECT="${PROJECT:-ogbench_cube_debug}"
DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/cube_single_4way_${TS}"
mkdir -p "${LOG_DIR}"

mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
NUM_GPUS="${#GPU_IDS[@]}"
RUN_IDX=0
PIDS=()

COMMON_ARGS=(
  --env_name cube-single-singletask-task1-v0
  --obs_mode state
  --num_envs "${NUM_ENVS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --learning_starts 2000
  --batch_size 512
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes 20
  --save_interval "${SAVE_INTERVAL}"
  --use_wandb
  --project "${PROJECT}"
)

launch_run() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local gpu=""
  if (( NUM_GPUS > 0 )); then
    gpu="${GPU_IDS[$((RUN_IDX % NUM_GPUS))]}"
  fi

  echo "Launching ${name} (gpu=${gpu:-cpu}) -> ${log_file}"

  if [[ -n "${gpu}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
    WANDB_MODE=online \
    WANDB_CONSOLE=off \
    WANDB_SILENT=true \
    python train_fast_sac_ogbench_manip.py \
      "${COMMON_ARGS[@]}" \
      --exp_name "${name}" \
      "$@" \
      > "${log_file}" 2>&1 &
  else
    WANDB_MODE=online \
    WANDB_CONSOLE=off \
    WANDB_SILENT=true \
    python train_fast_sac_ogbench_manip.py \
      "${COMMON_ARGS[@]}" \
      --exp_name "${name}" \
      "$@" \
      > "${log_file}" 2>&1 &
  fi

  PIDS+=("$!")
  RUN_IDX=$((RUN_IDX + 1))
}

# 1) Demo prefill (teacher) + dense reward
launch_run "cube_single_demo_dense_${TS}" \
  --reward_type dense \
  --cube_reward_mode legacy \
  --teacher_type cube_plan \
  --tolerance_type l2 \
  --tolerance_value 0.05 \
  --demo_buffer_enable \
  --demo_buffer_capacity 100000 \
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}" \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent \
  --demo_sample_ratio 0.5

# 2) Demo prefill (teacher) + sparse reward
launch_run "cube_single_demo_sparse_${TS}" \
  --reward_type sparse \
  --cube_reward_mode legacy \
  --teacher_type cube_plan \
  --tolerance_type l2 \
  --tolerance_value 0.05 \
  --demo_buffer_enable \
  --demo_buffer_capacity 100000 \
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}" \
  --demo_prefill_target demo \
  --demo_prefill_intervention_mode agent \
  --demo_sample_ratio 0.5

# 3) No demo prefill + dense reward
launch_run "cube_single_nodemo_dense_${TS}" \
  --reward_type dense \
  --cube_reward_mode legacy

# 4) No demo prefill + sparse reward
launch_run "cube_single_nodemo_sparse_${TS}" \
  --reward_type sparse \
  --cube_reward_mode legacy

echo
echo "Started PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}"
echo "Example monitor:"
echo "  tail -f ${LOG_DIR}/*.log"
echo
echo "Wait for completion:"
echo "  wait ${PIDS[*]}"
