#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/c/Data/diffusion-models-learning"
lerobot_root="$repo_root/repos/lerobot"
uv_bin="/home/benjamin/.local/bin/uv"

cd "$lerobot_root"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$repo_root/.cache/uv"
export UV_PROJECT_ENVIRONMENT="/home/benjamin/thesis/.venvs/smolvla-lerobot"
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=2
export HF_HOME="/home/benjamin/thesis/.cache/smolvla-hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"

echo "smolvla_setup_start"
"$uv_bin" sync --locked --extra smolvla --extra pusht --extra training --extra evaluation
"$uv_bin" run python -c 'import torch, transformers, lerobot, gym_pusht, av; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); print("transformers", transformers.__version__); print("lerobot", lerobot.__file__); print("gym_pusht", gym_pusht.__file__); print("av", av.__version__)'
echo "smolvla_setup_done"
