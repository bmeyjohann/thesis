#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" != *=* ]]; then
    echo "expected KEY=VALUE override, got: $arg" >&2
    exit 2
  fi
  export "$arg"
done

ROOT="${ROOT:-/home/benjamin/thesis}"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
CONTROLLER="${CONTROLLER:-scripted}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/logs/safetygym_eval_audits/native_teacher_${CONTROLLER}_${RUN_TAG}"

cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-safetygym-native-teacher}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$OUT"

"$PY" eval_interactive_safetygym.py \
  --controller "$CONTROLLER" \
  --render_mode none \
  --env_name SafetyCarGoal1-v0 \
  --layout_curriculum car_random_blocked_filter \
  --terminate_on_goal \
  --reward_mode dense \
  --dense_reward_scale 1.0 \
  --success_reward_scale 0.0 \
  --step_penalty 0.0 \
  --clearance_penalty_scale 0.0 \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --car_action_mode raw_wheels \
  --scale_actor_to_env_bounds \
  --use_layer_norm \
  --actor_hidden_dim 256 \
  --num_episodes "$EPISODES" \
  --seed "$SEED" \
  --save_episode_plots \
  --episode_plot_dir "$OUT" \
  --episode_plot_max_episodes 25 \
  --episode_plot_reward_surface | tee "$OUT/metrics.log"
