#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_prefill_aug}"
export STEPS="${STEPS:-120000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
export SEED="${SEED:-21}"

# Keep the learner reward clean: goal-distance progress only, no obstacle field reward.
export REWARD_MODE="${REWARD_MODE:-dense}"
export DENSE_REWARD_SCALE="${DENSE_REWARD_SCALE:-1.0}"
export SUCCESS_REWARD_SCALE="${SUCCESS_REWARD_SCALE:-0.0}"
export STEP_PENALTY="${STEP_PENALTY:-0.0}"
export CLEARANCE_PENALTY_SCALE="0.0"

# Collect a broad replay/pref warm start from the goal-only student under the
# same scripted-geo intervention gate, but do not use BC or demo sampling.
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-200}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"
export DEMO_SAMPLE_RATIO="0.0"
export DEMO_PRETRAIN_UPDATES="0"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.0}"

# Test whether local action-space augmentation improves preference transfer.
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
export PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.1}"
export PREF_ACTION_NOISE_COPIES="${PREF_ACTION_NOISE_COPIES:-2}"
export PREF_ACTION_NOISE_STD="${PREF_ACTION_NOISE_STD:-0.05}"
export PREF_OBS_NOISE_STD="${PREF_OBS_NOISE_STD:-0.0}"

export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_clearance_or_progress_long_20260529}"

exec "$BASE"
