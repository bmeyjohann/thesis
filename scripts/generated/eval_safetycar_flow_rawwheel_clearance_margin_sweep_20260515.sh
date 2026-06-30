#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/benjamin/thesis}"
PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
CHECKPOINT="${CHECKPOINT:-/home/benjamin/thesis/models/safetygym_flow/safetycar_flow_scriptedgeo_privgeom_dataset_goal1_randomblocked_20260515/step_10000.pt}"
OUT_ROOT="${OUT_ROOT:-/home/benjamin/thesis/logs/safetygym_flow/clearance_margin_sweep_20260515}"

mkdir -p "${OUT_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thesis_mpl_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

for spec in "m0.05_w0.75_g2.0" "m0.05_w1.5_g2.0" "m0.10_w0.75_g2.0" "m0.10_w1.5_g2.0"; do
  margin="$(echo "${spec}" | sed -E 's/^m([^_]+)_.*/\1/')"
  weight="$(echo "${spec}" | sed -E 's/.*_w([^_]+)_.*/\1/')"
  goal_weight="$(echo "${spec}" | sed -E 's/.*_g([^_]+)$/\1/')"
  out_dir="${OUT_ROOT}/${spec}"
  echo "=== eval margin=${margin} clearance_weight=${weight} goal_weight=${goal_weight} out=${out_dir} ==="
  "${PYTHON}" "${REPO}/train_flow_matching_safetygym.py" \
    --eval_only \
    --checkpoint_path "${CHECKPOINT}" \
    --device auto \
    --num_eval_episodes 8 \
    --eval_save_episode_plots \
    --eval_episode_plot_max_episodes 8 \
    --eval_output_dir "${out_dir}" \
    --eval_num_samples 8 \
    --sample_steps 8 \
    --sample_noise_scale 1.0 \
    --eval_chunk_selector best_goal_clearance \
    --eval_chunk_goal_weight "${goal_weight}" \
    --eval_chunk_clearance_weight "${weight}" \
    --eval_chunk_clearance_margin "${margin}" \
    --eval_chunk_rollout_step_scale 0.08 \
    --eval_chunk_rollout_turn_scale 0.35
done
