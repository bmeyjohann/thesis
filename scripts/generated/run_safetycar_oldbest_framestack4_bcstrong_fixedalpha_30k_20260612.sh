#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"

# Reproduce the prior best old-env thesis-method recipe at the same 30k
# budget as the competitor matrix. This is the scratch frame-stack-4,
# strong-BC + preference, fixed-alpha scripted-geo intervention setup that
# produced the earlier good 30k checkpoint.
export STEPS="${STEPS:-30000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-30000}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_oldbest_recovery_20260612}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_oldbest_framestack4_bcstrong_fixedalpha_30k}"
export SEED="${SEED:-65}"
export DISABLE_POLICY_VIZ="${DISABLE_POLICY_VIZ:-1}"

exec "$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh" \
  framestack_bcstrong_fixedalpha_scratch
