#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

exec /home/benjamin/thesis/scripts/run_safetygym_flow_matching_local.sh \
  EXP_NAME=safetycar_flow_scriptedgeo_privgeom_goal1_randomblocked_20260515 \
  EXP_PREFIX=safetycar_flow_scriptedgeo_privgeom_goal1_randomblocked \
  ENV_NAME=SafetyCarGoal1-v0 \
  WANDB_GROUP=safetycar_curriculum_flow_scriptedgeo_privgeom_20260515 \
  WANDB_MODE=online \
  PROJECT=thesis-safetygym \
  SEED=914 \
  TEACHER_MODE=scripted_geo \
  DATASET_STEPS=70000 \
  DATASET_LOG_EPISODES=25 \
  TRAIN_STEPS=40000 \
  BATCH_SIZE=512 \
  CHUNK_LEN=12 \
  HIDDEN_DIM=512 \
  DEPTH=4 \
  LEARNING_RATE=0.0003 \
  WEIGHT_DECAY=0.0001 \
  SAMPLE_STEPS=12 \
  EVAL_NUM_SAMPLES=16 \
  EVAL_CHUNK_SELECTOR=best_goal \
  REWARD_MODE=dense_plus_sparse \
  DENSE_REWARD_SCALE=1.0 \
  SUCCESS_REWARD_SCALE=5.0 \
  STEP_PENALTY=-0.001 \
  CLEARANCE_PENALTY_MODE=softplus \
  CLEARANCE_MARGIN=0.0 \
  CLEARANCE_PENALTY_SCALE=4.0 \
  CLEARANCE_PENALTY_TEMPERATURE=0.001 \
  CAR_ACTION_MODE=raw_wheels \
  CAR_WHEEL_COMMAND_LIMIT=1.0 \
  CAR_FORCE_SCALE=1.0 \
  OBS_MASK_MODE=privileged_geometry \
  MAX_EPISODE_STEPS=0 \
  LAYOUT_CURRICULUM=car_random_blocked_filter \
  LAYOUT_CURRICULUM_LEVEL=0 \
  NUM_EVAL_EPISODES=24 \
  EVAL_EPISODE_PLOT_MAX_EPISODES=12 \
  LOG_INTERVAL=1000 \
  EVAL_INTERVAL=5000 \
  SAVE_INTERVAL=10000
