#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/benjamin/miniconda3/envs/fasttd3/bin/python" ]]; then
    PYTHON_BIN="/home/benjamin/miniconda3/envs/fasttd3/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No usable python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 127
  fi
fi
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-ogbench-maze-state}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-}"
ENV_NAME="${ENV_NAME:-pointmaze-arena-danger-lethal-v0}"
ALGO_VARIANT_RAW="${ALGO_VARIANT:-own}"
ALGO_VARIANT="$(printf '%s' "${ALGO_VARIANT_RAW}" | tr '[:upper:]' '[:lower:]')"
MPLCONFIGDIR_VALUE="${MPLCONFIGDIR:-/tmp/matplotlib}"
WANDB_DIR_VALUE="${WANDB_DIR:-${PWD}/wandb}"
WANDB_CACHE_DIR_VALUE="${WANDB_CACHE_DIR:-${PWD}/.cache/wandb}"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

NUM_ENVS="${NUM_ENVS:-16}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-0}"
SEED="${SEED:-42}"
LEARNING_STARTS="${LEARNING_STARTS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_UPDATES="${NUM_UPDATES:-1}"
CTA_RATIO="${CTA_RATIO:-2}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
GAMMA="${GAMMA:-0.99}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"

EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
VIZ_GRID_RESOLUTION="${VIZ_GRID_RESOLUTION:-32}"
VIZ_QUIVER_STRIDE="${VIZ_QUIVER_STRIDE:-2}"
EVAL_SAVE_TRAJECTORY_PLOT="${EVAL_SAVE_TRAJECTORY_PLOT:-1}"
EVAL_TRAJECTORY_PLOT_EPISODES="${EVAL_TRAJECTORY_PLOT_EPISODES:-5}"
VIZ_ON_CHECKPOINT="${VIZ_ON_CHECKPOINT:-1}"
USE_LAYER_NORM="${USE_LAYER_NORM:-1}"

TEACHER_TYPE="${TEACHER_TYPE:-bfs}"
INTERVENTION_MODE="${INTERVENTION_MODE:-agent}"
TOLERANCE_TYPE="${TOLERANCE_TYPE:-angle}"
TOLERANCE_VALUE="${TOLERANCE_VALUE:-30.0}"
HARD_BLOCK_LETHAL="${HARD_BLOCK_LETHAL:-1}"
INTERVENTION_EPISODE_PROB="${INTERVENTION_EPISODE_PROB:-1.0}"
INTERVENTION_EPISODE_PROB_MIN="${INTERVENTION_EPISODE_PROB_MIN:-1.0}"
INTERVENTION_EPISODE_PROB_DECAY_STEPS="${INTERVENTION_EPISODE_PROB_DECAY_STEPS:-0}"
REWARD_TYPE="${REWARD_TYPE:-dense}"
DENSE_REWARD_SCALE="${DENSE_REWARD_SCALE:-1.0}"

DEMO_BUFFER_ENABLE="${DEMO_BUFFER_ENABLE:-0}"
DEMO_BUFFER_CAPACITY="${DEMO_BUFFER_CAPACITY:-200000}"
DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-20}"
DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-0}"
STORE_INTERVENED_IN_DEMO_BUFFER="${STORE_INTERVENED_IN_DEMO_BUFFER:-0}"

PREF_BUFFER_ENABLE="${PREF_BUFFER_ENABLE:-1}"
PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
PREF_SAMPLING_MODE="${PREF_SAMPLING_MODE:-linked}"
PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.1}"
PREF_CRITIC_SCOPE="${PREF_CRITIC_SCOPE:-all}"
PREF_LOSS_TYPE="${PREF_LOSS_TYPE:-lagrangian}"
PREF_LINKED_ACTION_EPSILON="${PREF_LINKED_ACTION_EPSILON:-0.000001}"
PREF_LINKED_ACTION_WEIGHT_SCALE="${PREF_LINKED_ACTION_WEIGHT_SCALE:-0.0}"
PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-0.0}"
PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.001}"
PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"
PREF_LAMBDA_EMA="${PREF_LAMBDA_EMA:-0.9}"
PREF_LAGRANGIAN_SCOPE="${PREF_LAGRANGIAN_SCOPE:-global}"
PREF_VIOLATION_CLIP="${PREF_VIOLATION_CLIP:-10.0}"
PREF_VIOLATION_TARGET="${PREF_VIOLATION_TARGET:-0.0}"
PREF_LAGRANGIAN_VIOLATION_TYPE="${PREF_LAGRANGIAN_VIOLATION_TYPE:-hinge}"
PREF_STOPGRAD_POSITIVE="${PREF_STOPGRAD_POSITIVE:-0}"

EIL_THRESHOLD="${EIL_THRESHOLD:-0.0}"
EIL_GOOD_MARGIN="${EIL_GOOD_MARGIN:-0.0}"
EIL_BAD_MARGIN="${EIL_BAD_MARGIN:-0.01}"
EIL_PAIR_MARGIN="${EIL_PAIR_MARGIN:-0.01}"
EIL_BAD_PRE_STEPS="${EIL_BAD_PRE_STEPS:-8}"

PVP_PROXY_VALUE_BOUND="${PVP_PROXY_VALUE_BOUND:-1.0}"
PVP_INCLUDE_ENV_REWARD_IN_TD="${PVP_INCLUDE_ENV_REWARD_IN_TD:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
ENV_TAG="${ENV_NAME//-/_}"
EXP_NAME="${EXP_NAME:-maze_${ALGO_VARIANT}_${ENV_TAG}_${TS}}"
LOG_FILE="logs/${EXP_NAME}.log"
mkdir -p logs
mkdir -p "${MPLCONFIGDIR_VALUE}" "${WANDB_DIR_VALUE}" "${WANDB_CACHE_DIR_VALUE}"

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --num_envs "${NUM_ENVS}"
  --max_episode_steps "${MAX_EPISODE_STEPS}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --seed "${SEED}"
  --device auto
  --train_render_mode none
  --obs_mode state
  --reward_type "${REWARD_TYPE}"
  --dense_reward_scale "${DENSE_REWARD_SCALE}"
  --use_intervention
  --intervention_mode "${INTERVENTION_MODE}"
  --teacher_type "${TEACHER_TYPE}"
  --tolerance_type "${TOLERANCE_TYPE}"
  --tolerance_value "${TOLERANCE_VALUE}"
  --intervention_episode_prob "${INTERVENTION_EPISODE_PROB}"
  --intervention_episode_prob_min "${INTERVENTION_EPISODE_PROB_MIN}"
  --intervention_episode_prob_decay_steps "${INTERVENTION_EPISODE_PROB_DECAY_STEPS}"
  --num_critics "${NUM_CRITICS}"
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}"
  --batch_size "${BATCH_SIZE}"
  --num_updates "${NUM_UPDATES}"
  --cta_ratio "${CTA_RATIO}"
  --learning_starts "${LEARNING_STARTS}"
  --gamma "${GAMMA}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --eval_num_envs "${EVAL_NUM_ENVS}"
  --eval_render_mode none
  --save_interval "${SAVE_INTERVAL}"
  --log_interval "${LOG_INTERVAL}"
  --project "${PROJECT}"
  --exp_name "${EXP_NAME}"
)

if [[ -n "${WANDB_ENTITY_VALUE}" ]]; then
  COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY_VALUE}")
fi
if [[ -n "${WANDB_GROUP_VALUE}" ]]; then
  COMMON_ARGS+=(--wandb_group "${WANDB_GROUP_VALUE}")
fi

if [[ "${HARD_BLOCK_LETHAL}" == "1" ]]; then
  COMMON_ARGS+=(--hard_block_lethal)
fi
if [[ "${USE_LAYER_NORM}" == "1" ]]; then
  COMMON_ARGS+=(--use_layer_norm)
fi
if [[ "${VIZ_ON_CHECKPOINT}" == "1" ]]; then
  COMMON_ARGS+=(--viz_on_checkpoint --viz_grid_resolution "${VIZ_GRID_RESOLUTION}" --viz_quiver_stride "${VIZ_QUIVER_STRIDE}")
fi
if [[ "${EVAL_SAVE_TRAJECTORY_PLOT}" == "1" ]]; then
  COMMON_ARGS+=(--eval_save_trajectory_plot --eval_trajectory_plot_episodes "${EVAL_TRAJECTORY_PLOT_EPISODES}")
fi

METHOD_ARGS=()
case "${ALGO_VARIANT}" in
  own)
    METHOD_ARGS+=(
      --pref_buffer_enable
      --pref_sample_ratio "${PREF_SAMPLE_RATIO}"
      --pref_sampling_mode "${PREF_SAMPLING_MODE}"
      --pref_rank_weight "${PREF_RANK_WEIGHT}"
      --pref_rank_margin "${PREF_RANK_MARGIN}"
      --pref_critic_scope "${PREF_CRITIC_SCOPE}"
      --pref_loss_type "${PREF_LOSS_TYPE}"
      --pref_linked_action_epsilon "${PREF_LINKED_ACTION_EPSILON}"
      --pref_linked_action_weight_scale "${PREF_LINKED_ACTION_WEIGHT_SCALE}"
      --pref_lambda_init "${PREF_LAMBDA_INIT}"
      --pref_lambda_lr "${PREF_LAMBDA_LR}"
      --pref_lambda_max "${PREF_LAMBDA_MAX}"
      --pref_lambda_ema "${PREF_LAMBDA_EMA}"
      --pref_lagrangian_scope "${PREF_LAGRANGIAN_SCOPE}"
      --pref_violation_clip "${PREF_VIOLATION_CLIP}"
      --pref_violation_target "${PREF_VIOLATION_TARGET}"
      --pref_lagrangian_violation_type "${PREF_LAGRANGIAN_VIOLATION_TYPE}"
    )
    if [[ "${PREF_STOPGRAD_POSITIVE}" == "1" ]]; then
      METHOD_ARGS+=(--pref_stopgrad_positive)
    fi
    if [[ "${DEMO_BUFFER_ENABLE}" == "1" ]]; then
      METHOD_ARGS+=(
        --demo_buffer_enable
        --demo_buffer_capacity "${DEMO_BUFFER_CAPACITY}"
        --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
      )
      if [[ "${STORE_INTERVENED_IN_DEMO_BUFFER}" == "1" ]]; then
        METHOD_ARGS+=(--store_intervened_in_demo_buffer)
      fi
      if [[ "${DEMO_PREFILL_EPISODES}" -gt 0 ]]; then
        METHOD_ARGS+=(
          --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
          --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
          --demo_prefill_target demo
          --demo_prefill_intervention_mode agent_always
        )
      fi
    fi
    ;;
  pvp)
    METHOD_ARGS+=(
      --algo_variant pvp
      --demo_buffer_enable
      --demo_buffer_capacity "${DEMO_BUFFER_CAPACITY}"
      --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
      --pvp_proxy_value_bound "${PVP_PROXY_VALUE_BOUND}"
    )
    if [[ "${PVP_INCLUDE_ENV_REWARD_IN_TD}" == "1" ]]; then
      METHOD_ARGS+=(--pvp_include_env_reward_in_td)
    fi
    if [[ "${DEMO_PREFILL_EPISODES}" -gt 0 ]]; then
      METHOD_ARGS+=(
        --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
        --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
        --demo_prefill_target demo
        --demo_prefill_intervention_mode agent_always
      )
    fi
    ;;
  eil)
    METHOD_ARGS+=(
      --algo_variant eil
      --eil_threshold "${EIL_THRESHOLD}"
      --eil_good_margin "${EIL_GOOD_MARGIN}"
      --eil_bad_margin "${EIL_BAD_MARGIN}"
      --eil_pair_margin "${EIL_PAIR_MARGIN}"
      --eil_bad_pre_steps "${EIL_BAD_PRE_STEPS}"
    )
    if [[ "${DEMO_BUFFER_ENABLE}" == "1" ]]; then
      METHOD_ARGS+=(
        --demo_buffer_enable
        --demo_buffer_capacity "${DEMO_BUFFER_CAPACITY}"
        --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
      )
      if [[ "${DEMO_PREFILL_EPISODES}" -gt 0 ]]; then
        METHOD_ARGS+=(
          --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
          --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
          --demo_prefill_target demo
          --demo_prefill_intervention_mode agent_always
        )
      fi
    fi
    ;;
  hilserl)
    METHOD_ARGS+=(
      --demo_buffer_enable
      --demo_buffer_capacity "${DEMO_BUFFER_CAPACITY}"
      --demo_sample_ratio "${DEMO_SAMPLE_RATIO}"
      --store_intervened_in_demo_buffer
      --pref_sample_ratio 0.0
      --pref_rank_weight 0.0
    )
    if [[ "${DEMO_PREFILL_EPISODES}" -gt 0 ]]; then
      METHOD_ARGS+=(
        --demo_prefill_episodes "${DEMO_PREFILL_EPISODES}"
        --demo_prefill_num_envs "${DEMO_PREFILL_NUM_ENVS}"
        --demo_prefill_target demo
        --demo_prefill_intervention_mode agent_always
      )
    fi
    ;;
  *)
    echo "Unsupported ALGO_VARIANT=${ALGO_VARIANT}. Use one of: own, pvp, eil, hilserl." >&2
    exit 1
    ;;
esac

echo "=== Starting ${EXP_NAME} (${ALGO_VARIANT}) ==="
echo "Log file: ${LOG_FILE}"

MPLCONFIGDIR="${MPLCONFIGDIR_VALUE}" \
WANDB_DIR="${WANDB_DIR_VALUE}" \
WANDB_CACHE_DIR="${WANDB_CACHE_DIR_VALUE}" \
WANDB_MODE="${WANDB_MODE_VALUE}" \
WANDB_CONSOLE=off \
WANDB_SILENT=true \
"${PYTHON_BIN}" train_fast_sac_ogbench_maze.py \
  "${COMMON_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  --use_wandb \
  2>&1 | tee "${LOG_FILE}"

echo "=== Finished ${EXP_NAME} ==="
