#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL="${MODEL:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"
AUDIT_ROOT="${AUDIT_ROOT:-$ROOT/logs/safetygym_eval_audits/visual_gate_sweep_20260603}"
EPISODES="${EPISODES:-50}"
PLOT_MAX_EPISODES="${PLOT_MAX_EPISODES:-12}"
SEED="${SEED:-10065}"
THRESHOLDS="${THRESHOLDS:-0.05 0.10 0.15 0.20}"

cd "$ROOT"
mkdir -p "$AUDIT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

for thr in $THRESHOLDS; do
  exit_thr="$("$PY" - "$thr" <<'PY'
import sys
thr = float(sys.argv[1])
print(f"{thr + 0.08:.3f}")
PY
)"
  label="$(printf 'thr%03d' "$(awk -v v="$thr" 'BEGIN { printf("%d", v * 1000) }')")"
  out_dir="$AUDIT_ROOT/${label}_exit${exit_thr}_50ep_seed${SEED}"
  mkdir -p "$out_dir"
  echo "[gate-audit] threshold=$thr exit=$exit_thr out=$out_dir"

  "$PY" "$ROOT/eval_interactive_safetygym.py" \
    --model_path "$MODEL" \
    --controller policy \
    --policy_format fastsac \
    --intervention_mode human \
    --human_input_device scripted_geo \
    --scripted_geo_heading_tolerance "${SCRIPTED_GEO_HEADING_TOLERANCE:-0.20}" \
    --scripted_geo_lookahead "${SCRIPTED_GEO_LOOKAHEAD:-1.4}" \
    --scripted_geo_safety_margin "${SCRIPTED_GEO_SAFETY_MARGIN:-0.35}" \
    --scripted_geo_grid_resolution "${SCRIPTED_GEO_GRID_RESOLUTION:-0.06}" \
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
    --teacher_clearance_source visual_footprint \
    --teacher_override_mode clearance_or_progress \
    --teacher_override_clearance_threshold "$thr" \
    --teacher_override_clearance_exit_threshold "$exit_thr" \
    --teacher_progress_score_mode euclidean \
    --teacher_progress_dense_scale 1.0 \
    --teacher_progress_trigger_mode not_improving \
    --teacher_progress_release_mode improve \
    --teacher_progress_bad_steps 3 \
    --teacher_progress_good_steps 5 \
    --teacher_progress_epsilon 0.0005 \
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
done
