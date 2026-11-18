#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
LOW_LEVEL="$ROOT/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712"

RUN_NAME="${RUN_NAME:-unitree_target_nav_curriculum_sac_seed0_80k_20260830}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2500}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
RANDOM_STEPS="${RANDOM_STEPS:-500}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unitree-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$ROOT"
exec "$PYTHON" -u "$ROOT/train_unitree_nav_thesis.py" \
  --task Unitree-G1-Nav-Obstacles-Safe-Collision \
  --method sac \
  --device cuda:0 \
  --seed 0 \
  --num-envs 4 \
  --total-steps "$TOTAL_STEPS" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-num-envs 4 \
  --eval-num-episodes 20 \
  --eval-seed 941 \
  --eval-timeout-s 900 \
  --eval-fail-fast \
  --eval-at-end \
  --output-dir "$ROOT/models/unitree_target_nav" \
  --run-name "$RUN_NAME" \
  --wandb-mode disabled \
  --low-level-policy-path "$LOW_LEVEL" \
  --target-terrain \
  --target-terrain-preset balanced \
  --target-terrain-seed 3 \
  --target-terrain-arena-size 24 \
  --target-terrain-material-resolution 1.0 \
  --target-terrain-start-boundary-margin 1.25 \
  --target-terrain-start-clearance 0.75 \
  --target-terrain-start-attempts 256 \
  --target-terrain-goal-boundary-margin 0.75 \
  --target-terrain-goal-clearance 0.75 \
  --target-terrain-goal-attempts 256 \
  --target-terrain-goal-feature-fraction 1.0 \
  --target-terrain-goal-curriculum \
  --target-terrain-goal-curriculum-steps 15000 \
  --target-terrain-goal-curriculum-start-min 1.5 \
  --target-terrain-goal-curriculum-start-max 3.0 \
  --target-terrain-goal-curriculum-start-feature-fraction 0.0 \
  --goal-distance-min 4.0 \
  --goal-distance-max 8.0 \
  --episode-length-s 20 \
  --navigation-episode-mode episodic \
  --success-dist 0.5 \
  --terminate-on-goal \
  --height-scan-resolution 0.25 \
  --height-scan-pattern grid \
  --height-scan-forward-size 5.0 \
  --height-scan-lateral-size 3.0 \
  --scan-history 1 \
  --action-history 20 \
  --goal-encoding distance_bearing \
  --goal-distance-scale 14.0 \
  --navigation-policy-hz 5.0 \
  --student-controller actor \
  --student-action-scale 1.0 \
  --intervention-gate-mode none \
  --teacher-warmup-steps 0 \
  --learner-reward-mode dense_progress \
  --dense-progress-scale 1.0 \
  --success-bonus 5.0 \
  --failure-penalty 0.0 \
  --fall-penalty -5.0 \
  --timeout-penalty -1.0 \
  --n-step 5 \
  --gamma 0.99 \
  --batch-size 256 \
  --replay-capacity 500000 \
  --learning-starts "$LEARNING_STARTS" \
  --random-steps "$RANDOM_STEPS" \
  --updates-per-step 4 \
  --policy-frequency 2 \
  --hidden-dim 256 \
  --use-layer-norm \
  --lr-actor 0.0003 \
  --lr-critic 0.0003 \
  --alpha-init 0.001 \
  --alpha-min 0.0 \
  --alpha-max 0.05 \
  --log-interval 250
