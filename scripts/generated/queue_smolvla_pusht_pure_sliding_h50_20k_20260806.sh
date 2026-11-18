#!/usr/bin/env bash
set -euo pipefail

root="/mnt/c/Data/diffusion-models-learning"
venv="/home/benjamin/thesis/.venvs/smolvla-lerobot"

export SLIDING_VLA_ASSET_ROOT="$root"
export SLIDING_VLA_ROOT="$root"
export SLIDING_VLA_OUTPUT_DIR="$root/questions/results/smolvla_local_4060_pure_sliding_h50_seed0_20k"

export SMOLVLA_PYTHON="$venv/bin/python"
export SMOLVLA_TRAIN_BIN="$venv/bin/lerobot-train"
export SMOLVLA_EVAL_BIN="$venv/bin/lerobot-eval"
export SMOLVLA_POLICY_PATH="lerobot/smolvla_base"
export SMOLVLA_VLM_MODEL_NAME="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
export SMOLVLA_DATASET_REPO_ID="lerobot/pusht"
export SMOLVLA_DATASET_ROOT="/home/benjamin/thesis/.cache/smolvla-hf-datasets/lerobot--pusht"

export HF_HOME="/home/benjamin/thesis/.cache/smolvla-hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SMOLVLA_NUM_PROCESSES=1
export SMOLVLA_BATCH_SIZE=1
export SMOLVLA_SEED=0
export SMOLVLA_VARIANTS=native-sliding
export SMOLVLA_NATIVE_TIME_MODE=rolling_ladder
export SMOLVLA_NATIVE_SLIDING_STEPS=20000
export SMOLVLA_CHUNK_SIZE=50
export SMOLVLA_ROLLING_BLOCK_SIZE=1
export SMOLVLA_EXECUTION_HORIZON=1
export SMOLVLA_CHUNK_EXECUTION_HORIZON=10
export SMOLVLA_FLOW_STEPS=10
export SMOLVLA_ROLLING_PRIME_BUFFER=true
export SMOLVLA_ROLLING_STABILIZE_PADDING=true
export SMOLVLA_SAVE_FREQ=1000
export SMOLVLA_WARMUP_STEPS=250
export SMOLVLA_EVAL_EPISODES=50
export SMOLVLA_EVAL_BATCH_SIZE=1
export SMOLVLA_EVAL_NATIVE_COLD=false
export SMOLVLA_EVAL_NATIVE_AS_CHUNK=true
export SMOLVLA_WANDB=false

if [[ -e "$SLIDING_VLA_OUTPUT_DIR" ]]; then
  echo "Output already exists: $SLIDING_VLA_OUTPUT_DIR" >&2
  exit 2
fi

exec "$root/cluster/juwels/runners/smolvla_pusht_ablation.sh"
