#!/usr/bin/env bash
set -euo pipefail

# Boundary smoke test for FastSAC cube state runs.
# Goal: force updates immediately and run long enough to cross at least one episode boundary.
#
# Usage:
#   conda activate fasttd3
#   bash scripts/smoke_fastsac_update_boundary.sh

cd "$(dirname "$0")/.."

python train_fast_sac_ogbench_manip.py \
  --env_name cube-triple-singletask-task5-v0 \
  --obs_mode state \
  --total_timesteps 1600 \
  --num_envs 8 \
  --learning_starts 0 \
  --log_interval 200 \
  --save_interval 0 \
  --eval_interval 0 \
  --use_intervention \
  --intervention_mode agent \
  --teacher_type cube_plan \
  --teacher_target_mode sequential \
  --cube_success_tolerance 0.04 \
  --tolerance_type l2 \
  --tolerance_value 0.02 \
  --intervention_episode_prob 1.0 \
  --intervention_episode_prob_min 1.0 \
  --intervention_episode_prob_decay_steps 0 \
  --intervention_episode_prob_decay_start 0 \
  --pref_buffer_enable \
  --pref_capacity 10000 \
  --pref_sample_ratio 0.3 \
  --pref_rank_margin 0.1 \
  --pref_rank_weight 1.0
