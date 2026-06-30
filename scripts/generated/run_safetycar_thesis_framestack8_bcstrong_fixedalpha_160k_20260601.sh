#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
export OBS_FRAME_STACK="${OBS_FRAME_STACK:-8}"
export TEMPORAL_ENCODER="${TEMPORAL_ENCODER:-none}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_framestack8_hybrid_bcstrong_fixedalpha_160k}"
exec /home/benjamin/thesis/scripts/generated/run_safetycar_thesis_long_followup_20260601.sh framestack_bcstrong_fixedalpha_scratch
