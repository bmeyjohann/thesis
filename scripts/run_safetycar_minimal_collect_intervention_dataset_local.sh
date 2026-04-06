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

export EXP_PREFIX="${EXP_PREFIX:-safetycar_min_collect_intervention_dataset}"
export VARIANT="${VARIANT:-plain}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
export MODULE_IMPL="${MODULE_IMPL:-custom}"
export REWARD_MODE="${REWARD_MODE:-dense}"
export GAMMA="${GAMMA:-0.99}"
export STEP_PENALTY="${STEP_PENALTY:--0.001}"
export COST_PENALTY="${COST_PENALTY:-0.0}"
export OBS_NORMALIZATION="${OBS_NORMALIZATION:-1}"
export SCALE_ACTOR_TO_ENV_BOUNDS="${SCALE_ACTOR_TO_ENV_BOUNDS:-1}"
export ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
export CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-1024}"
export CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
export CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
export CAR_ACTION_MODE="${CAR_ACTION_MODE:-raw_wheels}"
export RENDER_MODE="${RENDER_MODE:-human}"
export USE_INTERVENTION="${USE_INTERVENTION:-1}"
export HUMAN_INPUT_DEVICE="${HUMAN_INPUT_DEVICE:-gamepad}"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.0}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-0.0}"
export DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.0}"
export EXPORT_REPLAY_DATASET_INTERVAL="${EXPORT_REPLAY_DATASET_INTERVAL:-1000}"
export EXPORT_REPLAY_DATASET_LABEL="${EXPORT_REPLAY_DATASET_LABEL:-human_intervention_replay}"
export EXPORT_REPLAY_DATASET_DIR="${EXPORT_REPLAY_DATASET_DIR:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PROJECT="${PROJECT:-thesis-safetygym}"

"$ROOT/scripts/run_safetycar_minimal_human_ready_local.sh" "TIMESTAMP=$TIMESTAMP"
