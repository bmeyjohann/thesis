#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export WANDB_MODE=run
export WANDB_CONSOLE=on

ENV_ID=${ENV_ID:-pointmaze-arena-danger-lethal-v0}
TOTAL_STEPS=${TOTAL_STEPS:-20000}
NUM_ENVS=${NUM_ENVS:-8}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LOG_INTERVAL=${LOG_INTERVAL:-200}
PROJECT=${PROJECT:-ogbench_fast_sac_smoke}

INTERVENTION_ENABLE_AFTER_STEPS=${INTERVENTION_ENABLE_AFTER_STEPS:-10}
INTERVENTION_EP_PROB=${INTERVENTION_EP_PROB:-0.6}
INTERVENTION_EP_PROB_MIN=${INTERVENTION_EP_PROB_MIN:-0.2}
INTERVENTION_DECAY_START=${INTERVENTION_DECAY_START:-200}
INTERVENTION_DECAY_STEPS=${INTERVENTION_DECAY_STEPS:-2000}

VIZ_FIRST_STEP=${VIZ_FIRST_STEP:-5000}
VIZ_GRID_RES=${VIZ_GRID_RES:-16}
VIZ_QUIVER_STRIDE=${VIZ_QUIVER_STRIDE:-2}

python -u train_fast_sac_ogbench_maze.py \
  --env_name ${ENV_ID} \
  --obs_mode state \
  --reward_type sparse --dense_reward_scale 0.0 --step_penalty 0.0 \
  --total_timesteps ${TOTAL_STEPS} --num_envs ${NUM_ENVS} \
  --save_interval ${SAVE_INTERVAL} --log_interval ${LOG_INTERVAL} \
  --use_wandb --project ${PROJECT} \
  --use_intervention --intervention_mode agent_safety_progress --teacher_type bfs \
  --tolerance_type angle --tolerance_value 30 \
  --intervention_safety_margin_frac 0.25 --intervention_release_steps 3 \
  --intervention_enable_after_steps ${INTERVENTION_ENABLE_AFTER_STEPS} \
  --intervention_episode_prob ${INTERVENTION_EP_PROB} \
  --intervention_episode_prob_min ${INTERVENTION_EP_PROB_MIN} \
  --intervention_episode_prob_decay_start ${INTERVENTION_DECAY_START} \
  --intervention_episode_prob_decay_steps ${INTERVENTION_DECAY_STEPS} \
  --no_hard_block_lethal \
  --viz_on_checkpoint --viz_first_step ${VIZ_FIRST_STEP} \
  --viz_grid_resolution ${VIZ_GRID_RES} --viz_quiver_stride ${VIZ_QUIVER_STRIDE}
