#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PY=/home/benjamin/miniconda3/envs/fasttd3/bin/python
RUN_NAME="unitree_human_matched_sac_reward_only_seed0_28k_20260804"
OUT="$ROOT/models/unitree_mjlab_nav_thesis"
MANIFEST="$ROOT/local/eval_manifests/unitree_human_rectscan_blocked_100_seed0.json"

export MPLCONFIGDIR=/tmp/mplconfig
export WARP_CACHE_PATH=/tmp/warp-cache
export XDG_CACHE_HOME=/tmp/unitree-cache
export MUJOCO_GL=egl
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

echo "Matched Unitree reward-only SAC baseline"
echo "run:         $RUN_NAME"
echo "transitions: 28770 (one environment)"
echo "scanner:     5.0m forward x 3.0m lateral at 0.25m"
echo "reward:      dense Euclidean progress +20 success / -20 failure"
echo "safety:      costs logged only; no interventions, BC, or preference loss"
echo "eval:        fixed 100-layout human-run cohort"

exec "$PY" -u "$ROOT/train_unitree_nav_thesis.py" \
  --task Unitree-G1-Nav-Obstacles-Safe-Collision \
  --method sac \
  --device cuda:0 \
  --seed 0 \
  --num-envs 1 \
  --episode-length-s 16.0 \
  --low-level-policy-path "$ROOT/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712" \
  --output-dir "$OUT" \
  --run-name "$RUN_NAME" \
  --wandb-project thesis-unitree-nav-human \
  --wandb-group unitree_human_matched_reward_only_20260804 \
  --wandb-mode online \
  --total-steps 28770 \
  --replay-capacity 100000 \
  --learning-starts 500 \
  --random-steps 0 \
  --teacher-warmup-steps 0 \
  --batch-size 256 \
  --updates-per-step 1 \
  --policy-frequency 2 \
  --hidden-dim 256 \
  --policy-encoder mlp \
  --height-scan-resolution 0.25 \
  --height-scan-forward-size 5.0 \
  --height-scan-lateral-size 3.0 \
  --scan-history 1 \
  --action-history 0 \
  --student-action-smoothing 0.0 \
  --deterministic-student \
  --lr-actor 0.0003 \
  --lr-critic 0.0003 \
  --gamma 0.99 \
  --n-step 5 \
  --tau 0.005 \
  --max-grad-norm 10.0 \
  --alpha-init 0.01 \
  --alpha-min 0.01 \
  --alpha-max 0.01 \
  --actor-bc-weight 0.0 \
  --pref-rank-weight 0.0 \
  --intervention-gate-mode none \
  --learner-reward-mode dense_progress \
  --dense-progress-scale 1.0 \
  --success-bonus 20.0 \
  --failure-penalty -20.0 \
  --success-dist 0.4 \
  --terminate-on-goal \
  --goal-distance-min 4.5 \
  --goal-distance-max 8.0 \
  --min-goal-obstacle-clearance 0.9 \
  --goal-clearance-resample-attempts 100 \
  --min-start-obstacle-clearance 1.0 \
  --start-clearance-resample-attempts 100 \
  --require-blocked-corridor \
  --blocked-corridor-radius 0.45 \
  --blocked-corridor-ignore-end-radius 0.75 \
  --blocked-corridor-min-cells 4 \
  --blocked-corridor-resample-attempts 300 \
  --blocked-goal-max-distance 8.0 \
  --blocked-goal-distance-sampling uniform \
  --blocked-goal-placement-mode obstacle_multiplier \
  --blocked-goal-distance-multiplier-min 1.0 \
  --blocked-goal-distance-multiplier-max 2.0 \
  --blocked-goal-candidate-attempts 64 \
  --debug-obstacle-width-min 1.0 \
  --debug-obstacle-width-max 1.4 \
  --strict-min-size-obstacles \
  --debug-obstacle-height-min 1.0 \
  --debug-obstacle-height-max 1.0 \
  --debug-num-obstacles 6 \
  --debug-platform-width 2.0 \
  --debug-terrain-rows 5 \
  --debug-terrain-cols 10 \
  --debug-goal-through-obstacle \
  --goal-through-obstacle-prob 1.0 \
  --debug-goal-obstacle-min-dist 1.0 \
  --debug-goal-obstacle-max-dist 5.5 \
  --log-interval 100 \
  --checkpoint-interval 5000 \
  --eval-interval 0 \
  --eval-num-envs 20 \
  --eval-num-episodes 100 \
  --eval-seed 0 \
  --eval-layout-manifest "$MANIFEST" \
  --eval-at-end
