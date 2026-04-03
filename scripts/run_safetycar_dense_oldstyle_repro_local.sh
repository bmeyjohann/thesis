#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-150000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-1}"

RENDER_MODE="${RENDER_MODE:-none}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
REWARD_MODE="${REWARD_MODE:-dense}"
STEP_PENALTY="${STEP_PENALTY:-0.0}"

ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-1024}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-safetycar_goal2_dense_oldstyle_repro}"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_TAG}_${TS}}"
EXP_NAME="${EXP_NAME:-${WANDB_RUN_NAME}}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${CHECKPOINT_INTERVAL}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${CHECKPOINT_INTERVAL}}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"

VIZ_GRID_RESOLUTION="${VIZ_GRID_RESOLUTION:-24}"
VIZ_QUIVER_STRIDE="${VIZ_QUIVER_STRIDE:-4}"
VIZ_DEVICE="${VIZ_DEVICE:-cpu}"
VIZ_HEADINGS_DEG="${VIZ_HEADINGS_DEG:-0,90,180,270}"
VIZ_NUM_ROLLOUTS="${VIZ_NUM_ROLLOUTS:-3}"

echo "Starting SafetyCar old-style dense repro."
echo "Render mode: ${RENDER_MODE}"
echo "Reward mode: ${REWARD_MODE}"
echo "Actor/critic dims: ${ACTOR_HIDDEN_DIM}/${CRITIC_HIDDEN_DIM}"
echo "Wheel limit / force scale: ${CAR_WHEEL_COMMAND_LIMIT} / ${CAR_FORCE_SCALE}"
echo "Alpha clamp: [${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Eval interval: ${EVAL_INTERVAL}"
echo "Save interval: ${SAVE_INTERVAL}"
echo "Checkpoint policy maps enabled."

cmd=(
  conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" train_fast_sac_safetygym.py
  --env_name "${ENV_NAME}"
  --exp_name "${EXP_NAME}"
  --render_mode "${RENDER_MODE}"
  --surface_mode "${SURFACE_MODE}"
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}"
  --car_force_scale "${CAR_FORCE_SCALE}"
  --num_envs 1
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --update_every "${UPDATE_EVERY}"
  --updates_per_cycle "${UPDATES_PER_CYCLE}"
  --reward_mode "${REWARD_MODE}"
  --step_penalty "${STEP_PENALTY}"
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}"
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}"
  --num_critics "${NUM_CRITICS}"
  --alpha_min "${ALPHA_MIN}"
  --alpha_max "${ALPHA_MAX}"
  --pref_rank_weight 0.0
  --demo_sample_ratio 0.0
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --save_interval "${SAVE_INTERVAL}"
  --viz_on_checkpoint
  --viz_grid_resolution "${VIZ_GRID_RESOLUTION}"
  --viz_quiver_stride "${VIZ_QUIVER_STRIDE}"
  --viz_device "${VIZ_DEVICE}"
  --viz_headings_deg "${VIZ_HEADINGS_DEG}"
  --viz_num_rollouts "${VIZ_NUM_ROLLOUTS}"
  --eval_save_episode_plots
  --eval_episode_plot_max_episodes 9
  --use_wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_mode "${WANDB_MODE_VALUE}"
  --wandb_run_name "${WANDB_RUN_NAME}"
)

"${cmd[@]}"
