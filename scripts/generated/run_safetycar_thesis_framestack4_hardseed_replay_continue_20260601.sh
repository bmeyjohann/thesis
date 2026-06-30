#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_framestack4_best_continue100k_20260601.sh"

export STEPS="${STEPS:-30000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_hardseed_replay_20260601}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_hardseed_replay_continue_30k}"
export SEED="${SEED:-86}"

export LAYOUT_SEED_REPLAY="${LAYOUT_SEED_REPLAY:-$ROOT/local/safetygym_seed_replay_20260601/current_best_step30000_costful_seeds.txt}"
export LAYOUT_SEED_REPLAY_PROB="${LAYOUT_SEED_REPLAY_PROB:-0.5}"
export LAYOUT_SEED_REPLAY_MODE="${LAYOUT_SEED_REPLAY_MODE:-random}"
export EVAL_LAYOUT_SEED_REPLAY_PROB="${EVAL_LAYOUT_SEED_REPLAY_PROB:-0.0}"

exec "$BASE"
