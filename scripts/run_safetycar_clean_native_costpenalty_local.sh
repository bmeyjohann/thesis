#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*)
      export "$arg"
      ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-2}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-2}"

GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
ALPHA_INIT="${ALPHA_INIT:-1e-3}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"

ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-1024}"
NUM_CRITICS="${NUM_CRITICS:-2}"
CRITIC_LOSS_REDUCTION="${CRITIC_LOSS_REDUCTION:-sum}"

RENDER_MODE="${RENDER_MODE:-none}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
STEP_PENALTY="${STEP_PENALTY:--0.001}"
COST_PENALTY="${COST_PENALTY:-0.0}"

LOG_INTERVAL="${LOG_INTERVAL:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"

BASE_TAG="${BASE_TAG:-safetycar_clean_native_costpenalty}"
TS="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${BASE_TAG}_${TS}}"
WANDB_GROUP="${WANDB_GROUP:-}"

echo "Starting SafetyCar clean native-reward run."
echo "run_name=${RUN_NAME}"
echo "timesteps=${TOTAL_TIMESTEPS}"
echo "step_penalty=${STEP_PENALTY}"
echo "cost_penalty=${COST_PENALTY} (native reward already carries its own reward semantics; keep this at 0 unless you explicitly want extra collision shaping)"

cmd=(
  "${PYTHON_BIN}" /home/benjamin/thesis/train_fast_sac_safetygym.py
  --env_name "${ENV_NAME}"
  --exp_name "${RUN_NAME}"
  --render_mode "${RENDER_MODE}"
  --surface_mode "${SURFACE_MODE}"
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --update_every "${UPDATE_EVERY}"
  --updates_per_cycle "${UPDATES_PER_CYCLE}"
  --policy_frequency "${POLICY_FREQUENCY}"
  --gamma "${GAMMA}"
  --tau "${TAU}"
  --actor_learning_rate "${ACTOR_LR}"
  --critic_learning_rate "${CRITIC_LR}"
  --alpha_init "${ALPHA_INIT}"
  --alpha_min "${ALPHA_MIN}"
  --alpha_max "${ALPHA_MAX}"
  --critic_loss_reduction "${CRITIC_LOSS_REDUCTION}"
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}"
  --num_critics "${NUM_CRITICS}"
  --obs_normalization
  --reward_mode native
  --step_penalty "${STEP_PENALTY}"
  --cost_penalty "${COST_PENALTY}"
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}"
  --car_force_scale "${CAR_FORCE_SCALE}"
  --scale_actor_to_env_bounds
  --no_reseed_on_episode_reset
  --pref_capacity 0
  --pref_sample_ratio 0.0
  --pref_rank_weight 0.0
  --demo_sample_ratio 0.0
  --no_uncertainty_log_every_step
  --uncertainty_oversight_mode off
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${CHECKPOINT_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --save_interval "${CHECKPOINT_INTERVAL}"
  --viz_on_checkpoint
  --viz_grid_resolution 24
  --viz_quiver_stride 4
  --viz_device cpu
  --viz_headings_deg "0,90,180,270"
  --viz_num_rollouts 3
  --eval_save_episode_plots
  --eval_episode_plot_max_episodes 9
  --use_wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_mode "${WANDB_MODE_VALUE}"
)

if [[ -n "${WANDB_GROUP}" ]]; then
  cmd+=(--wandb_group "${WANDB_GROUP}")
fi

cmd+=(--wandb_run_name "${RUN_NAME}")

"${cmd[@]}"
