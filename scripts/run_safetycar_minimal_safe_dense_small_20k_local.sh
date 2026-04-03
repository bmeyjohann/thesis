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

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

EXP_PREFIX="${EXP_PREFIX:-safetycar_min_safe_dense_cost001_step_small_20k}"
export EXP_PREFIX
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
export MODULE_IMPL="${MODULE_IMPL:-custom}"
export REWARD_MODE="${REWARD_MODE:-dense}"
export GAMMA="${GAMMA:-0.99}"
export STEP_PENALTY="${STEP_PENALTY:--0.001}"
export COST_PENALTY="${COST_PENALTY:--0.01}"
export OBS_NORMALIZATION="${OBS_NORMALIZATION:-1}"
export SCALE_ACTOR_TO_ENV_BOUNDS="${SCALE_ACTOR_TO_ENV_BOUNDS:-1}"
export CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
export CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
export ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
export CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
export SEED="${SEED:-1}"
export RENDER_MODE="${RENDER_MODE:-none}"
export VIZ_ON_CHECKPOINT="${VIZ_ON_CHECKPOINT:-0}"
export EVAL_SAVE_EPISODE_PLOTS="${EVAL_SAVE_EPISODE_PLOTS:-0}"
export INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-}"
export LOAD_ACTOR_FROM_CHECKPOINT="${LOAD_ACTOR_FROM_CHECKPOINT:-1}"
export LOAD_CRITIC_FROM_CHECKPOINT="${LOAD_CRITIC_FROM_CHECKPOINT:-1}"
export LOAD_CRITIC_TARGET_FROM_CHECKPOINT="${LOAD_CRITIC_TARGET_FROM_CHECKPOINT:-1}"
export LOAD_ALPHA_FROM_CHECKPOINT="${LOAD_ALPHA_FROM_CHECKPOINT:-1}"
export LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT="${LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT="${PROJECT:-thesis-safetygym}"

"$ROOT/scripts/run_safetycar_minimal_human_ready_local.sh" "TIMESTAMP=$TIMESTAMP"
