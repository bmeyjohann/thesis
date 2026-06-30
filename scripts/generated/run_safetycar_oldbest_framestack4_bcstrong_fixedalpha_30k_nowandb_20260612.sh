#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"

# Same old-best recovery recipe as the online launcher, but disables W&B so the
# queue can still finish and write local metrics/checkpoints if W&B is unstable.
export STEPS="${STEPS:-30000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-30000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-30000}"
export DISABLE_WANDB=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_oldbest_recovery_20260612_nowandb}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_oldbest_framestack4_bcstrong_fixedalpha_30k_nowandb}"
export SEED="${SEED:-65}"
export DISABLE_POLICY_VIZ="${DISABLE_POLICY_VIZ:-1}"

exec "$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh" \
  framestack_bcstrong_fixedalpha_scratch
