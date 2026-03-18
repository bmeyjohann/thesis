#!/usr/bin/env bash
set -euo pipefail

# Runs 7 manipulation trainings sequentially overnight:
#   3 base configs x 2 intervention rates (100%, 50%) + 1 extra 100%-only variant
#
# Config A: baseline linked-pref lagrangian, UTD=5, pref_stopgrad_positive ON, 120k steps
# Config B: same as A but pref_stopgrad_positive OFF, 120k steps
# Config C: same as A but UTD=1, LayerNorm ON, 300k steps
# Config D (100%-only): same as A but batch_size=1024
#
# Optional overrides:
#   PROJECT=ogbench-manip-reward-debug \
#   TOTAL_TIMESTEPS_SHORT=120000 TOTAL_TIMESTEPS_LONG=300000 \
#   NUM_ENVS=32 EVAL_NUM_ENVS=1 BATCH_SIZE=256 \
#   bash scripts/run_cube_triple_linked_pref_overnight.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-reward-debug}"
ENV_NAME="${ENV_NAME:-cube-triple-singletask-task5-v0}"

NUM_CRITICS="${NUM_CRITICS:-5}"
NUM_ENVS="${NUM_ENVS:-32}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"

TOTAL_TIMESTEPS_SHORT="${TOTAL_TIMESTEPS_SHORT:-120000}"
TOTAL_TIMESTEPS_LONG="${TOTAL_TIMESTEPS_LONG:-300000}"

EVAL_INTERVAL="${EVAL_INTERVAL:-3000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-0.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-1e-3}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"

LAYER_NORM_EPS="${LAYER_NORM_EPS:-1e-5}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/overnight_cube_triple_linked_pref_${TS}"
mkdir -p "${LOG_DIR}"

if [[ "${SAVE_INTERVAL}" -lt 10000 ]]; then
  echo "[run_cube_triple_linked_pref_overnight] SAVE_INTERVAL=${SAVE_INTERVAL} is too low; clamping to 10000."
  SAVE_INTERVAL=10000
fi

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --obs_mode state
  --reward_type sparse
  --cube_reward_mode sparse_intermediate
  --include_relative_cube_features
  --num_critics "${NUM_CRITICS}"
  --num_envs "${NUM_ENVS}"
  --train_render_mode none
  --eval_render_mode none
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --use_intervention
  --intervention_mode agent_always
  --teacher_type cube_markov
  --intervention_episode_prob_decay_steps 0
  --pref_sampling_mode linked
  --pref_rank_weight "${PREF_RANK_WEIGHT}"
  --pref_loss_type lagrangian
  --pref_lambda_init "${PREF_LAMBDA_INIT}"
  --pref_lambda_lr "${PREF_LAMBDA_LR}"
  --pref_lambda_max "${PREF_LAMBDA_MAX}"
  --pref_lambda_ema "${PREF_LAMBDA_EMA}"
  --pref_violation_clip "${PREF_VIOLATION_CLIP}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --save_interval "${SAVE_INTERVAL}"
  --use_wandb
  --project "${PROJECT}"
)

run_one() {
  local exp_name="$1"
  shift
  local log_file="${LOG_DIR}/${exp_name}.log"
  echo "=== Starting ${exp_name} ==="
  WANDB_MODE="${WANDB_MODE_VALUE}" \
  WANDB_CONSOLE=off \
  WANDB_SILENT=true \
  "${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
    "${COMMON_ARGS[@]}" \
    --exp_name "${exp_name}" \
    "$@" \
    2>&1 | tee "${log_file}"
}

run_prob_suite() {
  local prob_label="$1"
  local prob_value="$2"

  # A) Baseline (UTD=5, stopgrad ON, 120k)
  run_one "cube_triple_linked_lagr_utd5_stopgrad_on_p${prob_label}_${TS}" \
    --total_timesteps "${TOTAL_TIMESTEPS_SHORT}" \
    --num_updates 5 \
    --intervention_episode_prob "${prob_value}" \
    --intervention_episode_prob_min "${prob_value}" \
    --pref_stopgrad_positive

  # B) Same as A, stopgrad OFF
  run_one "cube_triple_linked_lagr_utd5_stopgrad_off_p${prob_label}_${TS}" \
    --total_timesteps "${TOTAL_TIMESTEPS_SHORT}" \
    --num_updates 5 \
    --intervention_episode_prob "${prob_value}" \
    --intervention_episode_prob_min "${prob_value}"

  # C) UTD=1 + LayerNorm ON, 300k
  run_one "cube_triple_linked_lagr_utd1_ln_on_p${prob_label}_${TS}" \
    --total_timesteps "${TOTAL_TIMESTEPS_LONG}" \
    --num_updates 1 \
    --intervention_episode_prob "${prob_value}" \
    --intervention_episode_prob_min "${prob_value}" \
    --pref_stopgrad_positive \
    --use_layer_norm \
    --layer_norm_eps "${LAYER_NORM_EPS}"

  # D) 100%-only: UTD=5 + stopgrad ON + larger batch size
  if [[ "${prob_label}" == "100" ]]; then
    run_one "cube_triple_linked_lagr_utd5_stopgrad_on_bs1024_p${prob_label}_${TS}" \
      --total_timesteps "${TOTAL_TIMESTEPS_SHORT}" \
      --num_updates 5 \
      --batch_size 1024 \
      --intervention_episode_prob "${prob_value}" \
      --intervention_episode_prob_min "${prob_value}" \
      --pref_stopgrad_positive
  fi
}

echo "Logs directory: ${LOG_DIR}"
echo "=== Running 100% intervention suite ==="
run_prob_suite "100" "1.0"

echo "=== Running 50% intervention suite ==="
run_prob_suite "50" "0.5"

echo "=== All overnight runs completed successfully ==="
