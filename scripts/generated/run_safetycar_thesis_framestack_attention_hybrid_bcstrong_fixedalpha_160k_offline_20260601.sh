#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_attention_hybrid_bcstrong_fixedalpha_160k_offline}"
exec "$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh" framestack_attention_hybrid_scratch
