#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/c/Data/diffusion-models-learning"
venv="/home/benjamin/thesis/.venvs/smolvla-lerobot"
output="$repo_root/questions/results/smolvla_local_4060_smoke"

cd "$repo_root"
export HF_HOME="/home/benjamin/thesis/.cache/smolvla-hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin="$venv/bin/python"
train_bin="$venv/bin/lerobot-train"

[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
[[ -x "$train_bin" ]] || { echo "Missing trainer: $train_bin" >&2; exit 1; }

echo "smolvla_batch1_smoke_start"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
PYTHONPATH="$repo_root/repos/lerobot/src" "$python_bin" "$repo_root/tests/test_smolvla_rolling_port.py"

if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2
  exit 2
fi

"$train_bin" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=1 \
  --policy.num_steps=50 \
  --policy.rolling_block_size=1 \
  --policy.flow_time_mode=rolling_ladder \
  --policy.rolling_prime_buffer=true \
  --policy.rolling_stabilize_padding=true \
  --dataset.repo_id=lerobot/pusht \
  --dataset.root=/home/benjamin/thesis/.cache/smolvla-hf-datasets/lerobot--pusht \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --rename_map='{"observation.image":"observation.images.camera1"}' \
  --batch_size=1 \
  --num_workers=0 \
  --steps=1 \
  --log_freq=1 \
  --save_checkpoint=false \
  --eval_steps=0 \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$output" \
  --job_name=smolvla-local-4060-batch1-smoke

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "smolvla_batch1_smoke_done"
