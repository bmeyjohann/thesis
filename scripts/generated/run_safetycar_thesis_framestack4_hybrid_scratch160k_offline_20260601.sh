#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack4_hybrid_bc_pref_160k_offline}"
exec "$ROOT/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh" framestack_hybrid_scratch
