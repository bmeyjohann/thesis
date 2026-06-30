#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
export MUJOCO_GL="egl"

BASE="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_heading_richlive_throttle_150k_20260516.npz"
DAG1="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_heading_dagger_from_headingbc_richlive_throttle_50k_20260516.npz"
DAG2="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_heading_dagger2_from_headingbc_richlive_throttle_50k_20260516.npz"
WEIGHTED="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_heading_dagger_weighted_richlive_throttle_450k_20260516.npz"
OUT="/home/benjamin/thesis/models/safetygym_heading_bc/safetycar_heading_bc_scriptedgeo_richlive_throttle_dagger_weighted450k_20260516"
PLOT_DIR="/home/benjamin/thesis/logs/safetygym_heading_bc/posthoc_audited/heading_bc_scriptedgeo_richlive_throttle_dagger_weighted450k_seed424242_64ep_20260516"

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/merge_safetygym_heading_datasets.py \
  --output_path "${WEIGHTED}" \
  "${BASE}" \
  "${DAG1}" "${DAG1}" "${DAG1}" \
  "${DAG2}" "${DAG2}" "${DAG2}"

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_heading_bc_safetygym.py \
  --dataset_path "${WEIGHTED}" \
  --output_dir "${OUT}" \
  --seed 424242 \
  --device cuda \
  --epochs 160 \
  --batch_size 2048 \
  --lr 2e-4 \
  --weight_decay 3e-5 \
  --hidden_dims 512,512,256 \
  --activation tanh \
  --heading_tolerance 0.20 \
  --forward_throttle 0.8

mkdir -p "${PLOT_DIR}"
/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --controller policy \
  --policy_format heading_bc \
  --model_path "${OUT}/best.pt" \
  --env_name SafetyCarGoal1-v0 \
  --render_mode none \
  --seed 424242 \
  --num_episodes 64 \
  --reward_mode dense_plus_sparse \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --terminate_on_goal \
  --terminate_on_cost \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --obs_mask_mode privileged_geometry_rich \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --save_episode_plots \
  --plot_output_dir "${PLOT_DIR}" \
  --plot_max_episodes 16 \
  --plot_contact_sheet \
  --plot_reward_surface
