#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis/external/unitree_rl_mjlab

MPLCONFIGDIR=/tmp/mplconfig \
MUJOCO_GL=egl \
WANDB_MODE=online \
WARP_CACHE_PATH=/tmp/warp-cache \
XDG_CACHE_HOME=/tmp/unitree-cache \
/home/benjamin/miniconda3/envs/fasttd3/bin/python -u scripts/train.py \
  Unitree-G1-Flat-Omni \
  --env.scene.num-envs=2048 \
  --agent.resume=True \
  --agent.load-run=pretrained_model1499 \
  --agent.load-checkpoint='^model_1499[.]pt$' \
  --agent.max-iterations=30 \
  --agent.save-interval=10 \
  --agent.algorithm.learning-rate=0.0001 \
  --agent.algorithm.entropy-coef=0.001 \
  --agent.run-name=omni_finetune_model1499_20260712
