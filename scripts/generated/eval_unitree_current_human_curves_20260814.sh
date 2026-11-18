#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
FINAL_OUT="$ROOT/logs/unitree_human_hist1_action20_4run_paired100_20260813"
SWEEP_OUT="$ROOT/logs/unitree_current_objective_curves_20260814"
MANIFEST="$ROOT/logs/unitree_goalonly_sac_rect_hist3_action20_3seed_eval_20260813/layout_manifest_100_seed81330.json"

eval_policy() {
  local model_path="$1"
  local output_dir="$2"
  local run_name="$3"
  local episodes="$4"
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
    CONTROLLER=policy \
    MODEL_PATH="$model_path" \
    CHECKPOINT_ENV_CONFIG=1 \
    DEVICE=cuda:0 \
    SEED=81330 \
    NUM_ENVS=8 \
    NUM_EPISODES="$episodes" \
    EPISODE_LENGTH_S=90 \
    OUTPUT_DIR="$output_dir" \
    RUN_NAME="$run_name" \
    RECORD_VIDEO=0 \
    LAYOUT_MANIFEST="$MANIFEST"
}

if [[ "${1:-}" == "pref-only-final" ]]; then
  eval_policy \
    "$ROOT/models/unitree_mjlab_nav_human/unitree_human_pref_only_seed2_20260814_021008/final.pt" \
    "$FINAL_OUT" pref_only_seed2 100
  exit 0
fi

# Five matched checkpoints provide a readable teacher-free learning curve while
# keeping the additional evaluation tractable before the presentation.
steps=(1000 4000 7000 10000 13000)
declare -A runs=(
  [pref_bc_rl_seed1]="models/unitree_mjlab_nav_human/unitree_human_pref_bc_rl_seed1_20260813_142741"
  [pref_bc_rl_seed2]="models/unitree_mjlab_nav_human/unitree_human_pref_bc_rl_seed2_20260813_145006"
  [pref_rl_seed1]="models/unitree_mjlab_nav_human/unitree_human_pref_rl_seed1_20260813_150806"
  [pref_rl_seed2]="models/unitree_mjlab_nav_human/unitree_human_pref_rl_seed2_20260813_152522"
  [bc_rl_seed1]="models/unitree_mjlab_nav_human/unitree_human_bc_rl_seed1_20260813_181129"
  [bc_rl_seed2]="models/unitree_mjlab_nav_human/unitree_human_bc_rl_seed2_20260813_175038"
  [pref_only_seed1]="models/unitree_mjlab_nav_human/unitree_human_pref_only_seed1_20260814_011751"
  [pref_only_seed2]="models/unitree_mjlab_nav_human/unitree_human_pref_only_seed2_20260814_021008"
  [sac_seed1]="models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist1_action20_nofail_seed1_20260813_163309"
  [sac_seed2]="models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist1_action20_nofail_seed2_20260813_163309"
  [sac_seed3]="models/unitree_mjlab_nav_thesis/unitree_goalonly_sac_rect_hist1_action20_nofail_seed3_20260813_172054"
)

mkdir -p "$SWEEP_OUT"
if [[ -n "${ONLY_LABELS:-}" ]]; then read -r -a labels <<<"$ONLY_LABELS"; else labels=("${!runs[@]}"); fi
for label in "${labels[@]}"; do
  for step in "${steps[@]}"; do
    model="$ROOT/${runs[$label]}/step_${step}.pt"
    result="$SWEEP_OUT/${label}_step_${step}/policy_metrics.json"
    if [[ -s "$result" ]]; then
      echo "[curve-eval] skip existing $label step=$step"
      continue
    fi
    if [[ ! -s "$model" ]]; then
      echo "[curve-eval] missing $model" >&2
      continue
    fi
    echo "[curve-eval] $label step=$step"
    eval_policy "$model" "$SWEEP_OUT" "${label}_step_${step}" 20
  done
done
