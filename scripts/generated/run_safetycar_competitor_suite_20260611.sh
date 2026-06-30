#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" != *=* ]]; then
    echo "unexpected positional argument: $arg" >&2
    exit 2
  fi
  export "$arg"
done

ROOT="${ROOT:-/home/benjamin/thesis}"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-30000}"
SEED="${SEED:-211}"
METHODS="${METHODS:-pvp eil hilserl own}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/safetygym_competitor_comparison_${RUN_TS}}"
RAW_LOG_DIR="$OUT_ROOT/raw_logs"
PLOT_DIR="$OUT_ROOT/plots"
BASELINE_CKPT="${BASELINE_CKPT:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"

mkdir -p "$RAW_LOG_DIR" "$PLOT_DIR"
cd "$ROOT"

echo "[suite] methods=$METHODS steps=$STEPS seed=$SEED out=$OUT_ROOT"
if [[ "${INCLUDE_BASELINE:-1}" == "1" ]]; then
  baseline_log="$RAW_LOG_DIR/safetycar_goal1_comp_baseline_eval_seed${SEED}_${RUN_TS}.log"
  echo "[suite] running baseline eval log=$baseline_log"
  "$PY" "$ROOT/eval_interactive_safetygym.py" \
    --model_path "$BASELINE_CKPT" \
    --controller policy \
    --policy_format fastsac \
    --render_mode none \
    --env_name SafetyCarGoal1-v0 \
    --seed "$SEED" \
    --layout_curriculum car_random_blocked_filter \
    --terminate_on_goal \
    --reward_mode dense \
    --dense_reward_scale 1.0 \
    --success_reward_scale 0.0 \
    --step_penalty 0.0 \
    --clearance_penalty_scale 0.0 \
    --footprint_cost \
    --footprint_cost_mode visual \
    --footprint_cost_margin 0.0 \
    --footprint_cost_value 1.0 \
    --car_action_mode raw_wheels \
    --car_wheel_command_limit 2.0 \
    --car_force_scale 2.0 \
    --scale_actor_to_env_bounds \
    --use_layer_norm \
    --actor_hidden_dim 256 \
    --num_episodes "${BASELINE_EVAL_EPISODES:-${NUM_EVAL_EPISODES:-20}}" \
    --fps 0 \
    --no_show_episode_controls \
    --save_episode_plots \
    --episode_plot_dir "$OUT_ROOT/baseline_eval" \
    --episode_plot_max_episodes "${EVAL_EPISODE_PLOT_MAX_EPISODES:-9}" \
    2>&1 | tee "$baseline_log"
fi

for method in $METHODS; do
  exp_name="safetycar_goal1_comp_${method}_${STEPS}_seed${SEED}_${RUN_TS}"
  log_path="$RAW_LOG_DIR/${exp_name}.log"
  echo "[suite] running method=$method log=$log_path"
  METHOD="$method" \
  STEPS="$STEPS" \
  SEED="$SEED" \
  RUN_TS="$RUN_TS" \
  EXP_NAME="$exp_name" \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  WANDB_GROUP="${WANDB_GROUP:-safetycar_competitors_${RUN_TS}}" \
    "$ROOT/scripts/generated/run_safetycar_competitor_method_20260611.sh" \
    2>&1 | tee "$log_path"
done

"$PY" "$ROOT/tools/plot_safetygym_competitor_comparison.py" \
  --log_dir "$RAW_LOG_DIR" \
  --output_dir "$PLOT_DIR"

echo "[suite] wrote plots to $PLOT_DIR"
