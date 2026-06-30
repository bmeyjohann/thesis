#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"

export STEPS="${STEPS:-30000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_attention_probe_20260601}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_attention_probe_30k}"
export SEED="${SEED:-84}"
export OBS_FRAME_STACK="${OBS_FRAME_STACK:-4}"
export TEMPORAL_ENCODER="${TEMPORAL_ENCODER:-attention}"

exec "$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh" framestack_attention_hybrid_scratch
