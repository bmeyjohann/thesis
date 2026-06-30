#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

BASELINE_MODEL="${BASELINE_MODEL:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"
BEST_MODEL="${BEST_MODEL:-$ROOT/models/safetygym_minimal/safetycar_goal1_thesis_scriptedgeo_reward_160000_seed65_20260601_071229_framestack4_hybrid_bcstrong_fixedalpha_160k/step_30000.pt}"
AUDIT_ROOT="${AUDIT_ROOT:-$ROOT/logs/safetygym_eval_audits/visual_footprint_baseline_vs_best_20260603}"
EPISODES="${EPISODES:-100}"
PLOT_MAX_EPISODES="${PLOT_MAX_EPISODES:-25}"
SEED="${SEED:-10065}"
ENV_NAME="${ENV_NAME:-SafetyCarGoal1-v0}"

cd "$ROOT"
mkdir -p "$AUDIT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

run_policy_audit() {
  local name="$1"
  local model="$2"
  local out_dir="$AUDIT_ROOT/$name"
  mkdir -p "$out_dir"

  echo "[visual-audit] name=$name"
  echo "[visual-audit] model=$model"
  echo "[visual-audit] seed=$SEED episodes=$EPISODES out=$out_dir"

  "$PY" "$ROOT/eval_interactive_safetygym.py" \
    --model_path "$model" \
    --controller policy \
    --policy_format fastsac \
    --render_mode none \
    --env_name "$ENV_NAME" \
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
    --scale_actor_to_env_bounds \
    --use_layer_norm \
    --actor_hidden_dim 256 \
    --load_checkpoint_args \
    --num_episodes "$EPISODES" \
    --fps 0 \
    --save_episode_plots \
    --episode_plot_dir "$out_dir" \
    --episode_plot_max_episodes "$PLOT_MAX_EPISODES" \
    --episode_plot_reward_surface
}

run_policy_audit "goalonly_baseline_visual_footprint_100ep_seed${SEED}" "$BASELINE_MODEL"
run_policy_audit "thesis_best_framestack4_visual_footprint_100ep_seed${SEED}" "$BEST_MODEL"

