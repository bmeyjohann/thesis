#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_from_ppo50k_scriptedgeo_blocked_rich_20k_20260515.npz"
for _ in $(seq 1 240); do
  if [[ -s "$DATASET_PATH" ]]; then
    break
  fi
  echo "waiting for dataset: $DATASET_PATH"
  sleep 5
done
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "ERROR: dataset not found: $DATASET_PATH" >&2
  exit 2
fi

exec /home/benjamin/thesis/scripts/run_safetycar_minimal_human_ready_local.sh \
  EXP_NAME=safetycar_dagger_bc_from_ppo50k_rich_20kdata_cpu_20260515 \
  EXP_PREFIX=safetycar_dagger_bc_from_ppo50k_rich \
  VARIANT=own \
  ENV_NAME=SafetyCarGoal1-v0 \
  TOTAL_TIMESTEPS=30000 \
  CHECKPOINT_INTERVAL=5000 \
  LOG_INTERVAL=5000 \
  NUM_EVAL_EPISODES=32 \
  EVAL_SAVE_EPISODE_PLOTS=1 \
  EVAL_EPISODE_PLOT_MAX_EPISODES=16 \
  WANDB_MODE=online \
  PROJECT=thesis-safetygym \
  WANDB_GROUP=safetycar_scriptedgeo_dagger_ppo_source_20260515 \
  SEED=424242 \
  DEVICE=cpu \
  REWARD_MODE=dense_plus_sparse \
  DENSE_REWARD_SCALE=1.0 \
  SUCCESS_REWARD_SCALE=5.0 \
  STEP_PENALTY=-0.001 \
  CLEARANCE_PENALTY_MODE=softplus \
  CLEARANCE_MARGIN=0.0 \
  CLEARANCE_PENALTY_SCALE=4.0 \
  CLEARANCE_PENALTY_TEMPERATURE=0.001 \
  TERMINATE_ON_GOAL=1 \
  TERMINATE_ON_COST=1 \
  LAYOUT_CURRICULUM=car_random_blocked_filter \
  LAYOUT_CURRICULUM_LEVEL=0 \
  FIXED_LAYOUT_PRESET=none \
  CAR_WHEEL_COMMAND_LIMIT=1.0 \
  CAR_FORCE_SCALE=1.0 \
  CAR_ACTION_MODE=raw_wheels \
  OBS_MASK_MODE=privileged_geometry_rich \
  SCALE_ACTOR_TO_ENV_BOUNDS=1 \
  USE_LAYER_NORM=1 \
  ACTOR_HIDDEN_DIM=256 \
  CRITIC_HIDDEN_DIM=256 \
  BATCH_SIZE=512 \
  LEARNING_STARTS=0 \
  NUM_UPDATES=1 \
  POLICY_FREQUENCY=1 \
  ACTOR_LEARNING_RATE=0.0001 \
  CRITIC_LEARNING_RATE=0.0003 \
  ALPHA_INIT=0.001 \
  ACTOR_BC_WEIGHT=20.0 \
  ACTOR_BC_ONLY_UNTIL_STEP=30000 \
  DEMO_DATASET_PATH="$DATASET_PATH" \
  DEMO_DATASET_TARGET=variant \
  DEMO_DATASET_MAX_ROWS=0 \
  PREF_SAMPLE_RATIO=0.0 \
  PREF_RANK_WEIGHT=0.0 \
  PREF_CAPACITY=0 \
  DEMO_SAMPLE_RATIO=0.0 \
  PREFILL_DEMO_EPISODES=0 \
  OFFLINE_ONLY=1 \
  USE_INTERVENTION=0
