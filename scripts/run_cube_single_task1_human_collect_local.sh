#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LOCAL_DATASET_DIR="${REPO_ROOT}/local/ogbench_manip_datasets"

# Allow KEY=VALUE overrides so this launcher can be reused from the queue.
for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-manip-human-data}"
ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"
VR_HOST="${VR_HOST:-192.168.2.182}"
VR_PORT="${VR_PORT:-8765}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"
CTA_RATIO="${CTA_RATIO:-2}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
GAMMA="${GAMMA:-0.97}"

ALPHA_INIT="${ALPHA_INIT:-0.001}"
FIXED_ALPHA="${FIXED_ALPHA:-0.001}"
ALPHA_MIN="${ALPHA_MIN:-0.001}"
ALPHA_MAX="${ALPHA_MAX:-0.001}"
ALPHA_FREEZE_STEPS="${ALPHA_FREEZE_STEPS:-0}"

DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-0}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-0}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
STORE_INTERVENED_IN_DEMO_BUFFER="${STORE_INTERVENED_IN_DEMO_BUFFER:-0}"
DEMO_DATASET_PATH="${DEMO_DATASET_PATH:-}"
DEMO_DATASET_DIR="${DEMO_DATASET_DIR:-${DEFAULT_LOCAL_DATASET_DIR}}"
DEMO_DATASET_AUTO_LOAD="${DEMO_DATASET_AUTO_LOAD:-0}"
DEMO_DATASET_TARGET="${DEMO_DATASET_TARGET:-demo}"
DEMO_DATASET_MAX_ROWS="${DEMO_DATASET_MAX_ROWS:-0}"

INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-1.0}"
WAIT_FOR_HUMAN_START="${WAIT_FOR_HUMAN_START:-1}"

PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.01}"
PREF_CRITIC_SCOPE="${PREF_CRITIC_SCOPE:-all}"
PREF_LOSS_TYPE="${PREF_LOSS_TYPE:-lagrangian}"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-1e-3}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"
PREF_VIOLATION_TARGET="${PREF_VIOLATION_TARGET:-0.0}"
PREF_LAGRANGIAN_VIOLATION_TYPE="${PREF_LAGRANGIAN_VIOLATION_TYPE:-hinge}"

EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-5}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"

EXPORT_REPLAY_DATASET_INTERVAL="${EXPORT_REPLAY_DATASET_INTERVAL:-1000}"
EXPORT_REPLAY_DATASET_PATH="${EXPORT_REPLAY_DATASET_PATH:-}"
EXPORT_REPLAY_DATASET_DIR="${EXPORT_REPLAY_DATASET_DIR:-${DEFAULT_LOCAL_DATASET_DIR}}"
EXPORT_REPLAY_DATASET_LABEL="${EXPORT_REPLAY_DATASET_LABEL:-human_vr_online_replay}"
EXPORT_REPLAY_DATASET_MAX_ROWS="${EXPORT_REPLAY_DATASET_MAX_ROWS:-0}"
VISUALIZE_INTERVENTION_COLORS="${VISUALIZE_INTERVENTION_COLORS:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-cube_single_task1_human_collect_norot_fixedalpha1e3_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs

echo "=== Starting ${EXP_NAME} ==="
echo "Log file: ${LOG_FILE}"
echo "WANDB mode: ${WANDB_MODE_VALUE}"
echo "VR endpoint: ${VR_HOST}:${VR_PORT}"
echo "Replay dataset snapshots every ${EXPORT_REPLAY_DATASET_INTERVAL} env steps"
echo "Viewer intervention colors: ${VISUALIZE_INTERVENTION_COLORS}"
echo "Wait for human start trigger: ${WAIT_FOR_HUMAN_START}"
if [[ -n "${DEMO_DATASET_PATH}" ]]; then
  echo "Offline demo dataset path: ${DEMO_DATASET_PATH}"
elif [[ "${DEMO_DATASET_AUTO_LOAD}" == "1" ]]; then
  echo "Offline demo dataset auto-load dir: ${DEMO_DATASET_DIR}"
fi
echo "Store online interventions in demo buffer: ${STORE_INTERVENED_IN_DEMO_BUFFER}"
if [[ -n "${EXPORT_REPLAY_DATASET_PATH}" ]]; then
  echo "Replay dataset path: ${EXPORT_REPLAY_DATASET_PATH}"
else
  echo "Replay dataset dir: ${EXPORT_REPLAY_DATASET_DIR}"
fi
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Fixed alpha: ${FIXED_ALPHA}"

ARGS=(
  --env_name "${ENV_NAME}"
  --num_envs "${NUM_ENVS}"
  --max_episode_steps "${MAX_EPISODE_STEPS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --device auto
  --train_render_mode human
  --obs_mode state
  --no_include_goal
  --include_relative_cube_features
  --relative_only_obs
  --reward_type sparse
  --cube_reward_mode dense
  --disable_rotation
  --use_intervention
  --intervention_mode human
  --human_input_device vr
  --wait_for_human_start
  --vr_mode connect
  --vr_host "${VR_HOST}"
  --vr_port "${VR_PORT}"
  --teacher_type cube_markov
  --tolerance_type angle
  --tolerance_value 30.0
  --hard_block_lethal
  --num_critics "${NUM_CRITICS}"
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}"
  --batch_size "${BATCH_SIZE}"
  --num_updates "${NUM_UPDATES}"
  --cta_ratio "${CTA_RATIO}"
  --learning_starts "${LEARNING_STARTS}"
  --gamma "${GAMMA}"
  --alpha_init "${ALPHA_INIT}"
  --fixed_alpha "${FIXED_ALPHA}"
  --alpha_min "${ALPHA_MIN}"
  --alpha_max "${ALPHA_MAX}"
  --alpha_freeze_steps "${ALPHA_FREEZE_STEPS}"
  --use_layer_norm
  --demo_buffer_enable
  --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
  --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
  --demo_prefill_target demo
  --demo_prefill_intervention_mode agent_always
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
  --demo_dataset_target "${DEMO_DATASET_TARGET}"
  --pref_buffer_enable
  --pref_sampling_mode linked
  --pref_sample_ratio 0.0
  --pref_rank_weight "${PREF_RANK_WEIGHT}"
  --pref_rank_margin "${PREF_RANK_MARGIN}"
  --pref_critic_scope "${PREF_CRITIC_SCOPE}"
  --pref_loss_type "${PREF_LOSS_TYPE}"
  --pref_lambda_init "${PREF_LAMBDA_INIT}"
  --pref_lambda_lr "${PREF_LAMBDA_LR}"
  --pref_lambda_max "${PREF_LAMBDA_MAX}"
  --pref_lambda_ema "${PREF_LAMBDA_EMA}"
  --pref_violation_clip "${PREF_VIOLATION_CLIP}"
  --pref_violation_target "${PREF_VIOLATION_TARGET}"
  --pref_lagrangian_violation_type "${PREF_LAGRANGIAN_VIOLATION_TYPE}"
  --pref_stopgrad_positive
  --intervention_episode_prob "${INTERVENTION_EPISODE_PROB}"
  --intervention_episode_prob_min "${INTERVENTION_EPISODE_PROB}"
  --intervention_episode_prob_decay_steps 0
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --eval_render_mode none
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --use_wandb
  --project "${PROJECT}"
  --exp_name "${EXP_NAME}"
  --export_replay_dataset_interval "${EXPORT_REPLAY_DATASET_INTERVAL}"
  --export_replay_dataset_label "${EXPORT_REPLAY_DATASET_LABEL}"
  --tolerance_adaptive_near_distance 0.08
  --tolerance_adaptive_far_distance 0.30
  --tolerance_adaptive_near_scale 0.35
)

if [[ "${VISUALIZE_INTERVENTION_COLORS}" == "0" ]]; then
  ARGS+=(--no_visualize_intervention_colors)
fi

if [[ "${WAIT_FOR_HUMAN_START}" == "0" ]]; then
  ARGS+=(--no_wait_for_human_start)
fi

if [[ -n "${EXPORT_REPLAY_DATASET_PATH}" ]]; then
  ARGS+=(--export_replay_dataset_path "${EXPORT_REPLAY_DATASET_PATH}")
fi

if [[ -n "${DEMO_DATASET_PATH}" ]]; then
  ARGS+=(--demo_dataset_path "${DEMO_DATASET_PATH}")
fi

if [[ -n "${DEMO_DATASET_DIR}" ]]; then
  ARGS+=(--demo_dataset_dir "${DEMO_DATASET_DIR}")
fi

if [[ "${DEMO_DATASET_AUTO_LOAD}" == "1" ]]; then
  ARGS+=(--demo_dataset_auto_load)
fi

if [[ "${DEMO_DATASET_MAX_ROWS}" != "0" ]]; then
  ARGS+=(--demo_dataset_max_rows "${DEMO_DATASET_MAX_ROWS}")
fi

if [[ "${STORE_INTERVENED_IN_DEMO_BUFFER}" == "1" ]]; then
  ARGS+=(--store_intervened_in_demo_buffer)
fi

if [[ -n "${EXPORT_REPLAY_DATASET_DIR}" ]]; then
  ARGS+=(--export_replay_dataset_dir "${EXPORT_REPLAY_DATASET_DIR}")
fi

if [[ "${EXPORT_REPLAY_DATASET_MAX_ROWS}" != "0" ]]; then
  ARGS+=(--export_replay_dataset_max_rows "${EXPORT_REPLAY_DATASET_MAX_ROWS}")
fi

WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_manip.py \
  "${ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
