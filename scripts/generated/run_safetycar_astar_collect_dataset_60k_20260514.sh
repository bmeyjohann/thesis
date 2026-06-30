#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_astar_blocked_60k_evalfix_20260514.npz"
rm -f "$DATASET_PATH"

exec /home/benjamin/thesis/scripts/run_safetycar_minimal_human_ready_local.sh \
  EXP_NAME=safetycar_astar_collect_blocked_60k_evalfix_20260514 \
  EXP_PREFIX=safetycar_astar_collect_blocked_60k_evalfix \
  VARIANT=own \
  ENV_NAME=SafetyCarGoal1-v0 \
  TOTAL_TIMESTEPS=60000 \
  CHECKPOINT_INTERVAL=60000 \
  LOG_INTERVAL=10000 \
  NUM_EVAL_EPISODES=24 \
  EVAL_SAVE_EPISODE_PLOTS=1 \
  EVAL_EPISODE_PLOT_MAX_EPISODES=12 \
  WANDB_MODE=online \
  PROJECT=thesis-safetygym \
  WANDB_GROUP=safetycar_curriculum_astar_dataset_20260514 \
  SEED=170 \
  REWARD_MODE=dense_plus_sparse \
  DENSE_REWARD_SCALE=1.0 \
  SUCCESS_REWARD_SCALE=5.0 \
  STEP_PENALTY=-0.001 \
  CLEARANCE_PENALTY_MODE=softplus \
  CLEARANCE_MARGIN=0.0 \
  CLEARANCE_PENALTY_SCALE=4.0 \
  CLEARANCE_PENALTY_TEMPERATURE=0.001 \
  TERMINATE_ON_GOAL=1 \
  LAYOUT_CURRICULUM=car_random_blocked_filter \
  LAYOUT_CURRICULUM_LEVEL=0 \
  CAR_WHEEL_COMMAND_LIMIT=1.0 \
  CAR_FORCE_SCALE=1.0 \
  USE_LAYER_NORM=1 \
  ACTOR_HIDDEN_DIM=256 \
  CRITIC_HIDDEN_DIM=256 \
  LEARNING_STARTS=999999 \
  NUM_UPDATES=1 \
  POLICY_FREQUENCY=1 \
  USE_INTERVENTION=1 \
  HUMAN_INPUT_DEVICE=scripted_geo \
  INTERVENTION_THRESHOLD=0.0 \
  INTERVENTION_HOLD_SECONDS=0.0 \
  PREF_SAMPLE_RATIO=0.0 \
  PREF_RANK_WEIGHT=0.0 \
  PREF_CAPACITY=0 \
  DEMO_SAMPLE_RATIO=0.0 \
  PREFILL_DEMO_EPISODES=0 \
  EXPORT_FINAL_REPLAY_DATASET=1 \
  EXPORT_FINAL_REPLAY_DATASET_PATH="$DATASET_PATH"
