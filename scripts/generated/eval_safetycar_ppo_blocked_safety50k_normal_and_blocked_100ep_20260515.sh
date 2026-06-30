#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL="${MODEL:-/home/benjamin/thesis/models/safetygym_ppo/safetycar_ppo_warm50k_blocked_safety_20260515_104639/ppo_step_50000_steps.zip}"
OUT_ROOT="${OUT_ROOT:-/home/benjamin/thesis/logs/safetygym_ppo/posthoc_blocked_safety50k_audit_20260515}"

mkdir -p "${OUT_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thesis_mpl_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

echo "=== normal random 100 episode audit ==="
"${PYTHON}" /home/benjamin/thesis/scripts/eval_safetygym_ppo_curriculum_artifacts.py \
  --model_path "${MODEL}" \
  --output_dir "${OUT_ROOT}/normal_random_100ep" \
  --step_label 50000 \
  --num_episodes 100 \
  --seed 101 \
  --eval_fixed_layout_preset train \
  --eval_layout_curriculum none \
  --eval_layout_curriculum_level -1 \
  --plot_max_episodes 12

echo "=== random blocked filter 100 episode audit ==="
"${PYTHON}" /home/benjamin/thesis/scripts/eval_safetygym_ppo_curriculum_artifacts.py \
  --model_path "${MODEL}" \
  --output_dir "${OUT_ROOT}/random_blocked_filter_100ep" \
  --step_label 50000 \
  --num_episodes 100 \
  --seed 101 \
  --eval_fixed_layout_preset train \
  --eval_layout_curriculum car_random_blocked_filter \
  --eval_layout_curriculum_level 0 \
  --plot_max_episodes 12
