#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_fast_sac_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name safetycar_fastsac_scriptedgeo_hil_bc_probe_nowandb_12k_20260515 \
  --variant hilserl \
  --seed 858585 \
  --device cpu \
  --total_timesteps 12000 \
  --learning_starts 1000 \
  --batch_size 256 \
  --num_updates 1 \
  --policy_frequency 2 \
  --gamma 0.99 \
  --actor_learning_rate 3e-4 \
  --critic_learning_rate 3e-4 \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 1024 \
  --module_impl fastsac \
  --use_layer_norm \
  --actor_bc_weight 1.0 \
  --actor_bc_teacher_only \
  --demo_sample_ratio 0.5 \
  --store_intervened_in_demo_buffer \
  --reward_mode dense_plus_sparse \
  --dense_reward_scale 1.0 \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --clearance_penalty_mode softplus \
  --clearance_margin 0.0 \
  --clearance_penalty_scale 4.0 \
  --clearance_penalty_temperature 0.001 \
  --terminate_on_goal \
  --terminate_on_cost \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --obs_mask_mode privileged_geometry_rich \
  --car_action_mode raw_wheels \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --use_intervention \
  --human_input_device scripted_geo \
  --intervention_threshold 0.0 \
  --intervention_hold_seconds 0.0 \
  --scale_actor_to_env_bounds \
  --eval_interval 4000 \
  --num_eval_episodes 24 \
  --eval_save_episode_plots \
  --eval_episode_plot_max_episodes 12 \
  --save_interval 4000 \
  --log_interval 2000
