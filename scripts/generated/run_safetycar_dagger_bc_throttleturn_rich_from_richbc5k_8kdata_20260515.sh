#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_rich_from_richbc5k_8k_20260515.npz"
INIT="/home/benjamin/thesis/models/safetygym_minimal/safetycar_offline_scriptedgeo_bc_throttleturn_rich_scratch_8kdata_20260515/step_5000.pt"
for _ in $(seq 1 120); do
  if [[ -s "$DATASET_PATH" ]]; then
    break
  fi
  echo "waiting for dataset: $DATASET_PATH"
  sleep 30
done
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "ERROR: dataset not found after waiting: $DATASET_PATH" >&2
  exit 2
fi

exec /home/benjamin/thesis/scripts/run_safetycar_minimal_human_ready_local.sh \
  EXP_NAME=safetycar_dagger_bc_throttleturn_rich_from_richbc5k_8kdata_20260515 \
  EXP_PREFIX=safetycar_dagger_bc_throttleturn_rich_from_richbc5k_8kdata \
  VARIANT=own \
  ENV_NAME=SafetyCarGoal1-v0 \
  TOTAL_TIMESTEPS=20000 \
  CHECKPOINT_INTERVAL=5000 \
  LOG_INTERVAL=5000 \
  NUM_EVAL_EPISODES=24 \
  EVAL_SAVE_EPISODE_PLOTS=1 \
  EVAL_EPISODE_PLOT_MAX_EPISODES=12 \
  WANDB_MODE=online \
  PROJECT=thesis-safetygym \
  WANDB_GROUP=safetycar_curriculum_throttleturn_rich_daggerbc_20260515 \
  SEED=916 \
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
  OBS_MASK_MODE=privileged_geometry_rich \
  CAR_ACTION_MODE=throttle_turn \
  CAR_WHEEL_COMMAND_LIMIT=1.0 \
  CAR_FORCE_SCALE=1.0 \
  USE_LAYER_NORM=1 \
  ACTOR_HIDDEN_DIM=256 \
  CRITIC_HIDDEN_DIM=256 \
  BATCH_SIZE=512 \
  LEARNING_STARTS=0 \
  NUM_UPDATES=1 \
  POLICY_FREQUENCY=1 \
  ACTOR_LEARNING_RATE=0.0003 \
  CRITIC_LEARNING_RATE=0.0003 \
  ACTOR_BC_WEIGHT=10.0 \
  ACTOR_BC_ONLY_UNTIL_STEP=20000 \
  INIT_CHECKPOINT_PATH="$INIT" \
  LOAD_OPTIMIZER_STATE=0 \
  PREF_SAMPLE_RATIO=0.0 \
  PREF_RANK_WEIGHT=0.0 \
  PREF_CAPACITY=0 \
  DEMO_SAMPLE_RATIO=0.0 \
  PREFILL_DEMO_EPISODES=0 \
  DEMO_DATASET_PATH="$DATASET_PATH" \
  DEMO_DATASET_TARGET=variant \
  DEMO_DATASET_MAX_ROWS=0 \
  OFFLINE_ONLY=1
