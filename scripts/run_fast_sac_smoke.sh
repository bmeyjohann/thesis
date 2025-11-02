#!/usr/bin/env bash
# Quick smoke test for FastSAC OGBench training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python train_fast_sac_ogbench.py \
  --env_name pointmaze-arena-danger-lethal-v0 \
  --num_envs 8 \
  --total_timesteps 20000 \
  --batch_size 4096 \
  --buffer_size $((1024 * 20)) \
  --save_interval 0 \
  --log_interval 250 \
  --reward_type sparse \
  --use_intervention \
  --intervention_mode agent \
  --tolerance_value 30.0 \
  --pref_buffer_enable \
  --pref_buffer_npairs_per_add 2 \
  --pref_td_buffer_enable \
  --pref_td_sample_ratio 0.25 \
  --viz_on_checkpoint \
  "$@"
