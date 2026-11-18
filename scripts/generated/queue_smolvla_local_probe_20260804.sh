#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/c/Data/diffusion-models-learning"
cd "$repo_root"

echo "smolvla_local_probe_start"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
python -c 'import sys, importlib.util, torch; print("python", sys.executable); print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); print("transformers", bool(importlib.util.find_spec("transformers"))); print("lerobot", bool(importlib.util.find_spec("lerobot"))); print("gym_pusht", bool(importlib.util.find_spec("gym_pusht"))); print("av", bool(importlib.util.find_spec("av")))'

for path in \
  "$repo_root/repos/lerobot/.venv/bin/python" \
  "/home/benjamin/miniconda3/envs/fasttd3/bin/python" \
  "/home/benjamin/miniconda3/envs/lerobot/bin/python" \
  "/home/benjamin/miniconda3/envs/smolvla/bin/python"
do
  if [[ -x "$path" ]]; then
    echo "python_candidate=$path"
    "$path" -c 'import importlib.util, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", bool(importlib.util.find_spec("transformers")), "lerobot", bool(importlib.util.find_spec("lerobot")), "gym_pusht", bool(importlib.util.find_spec("gym_pusht")), "av", bool(importlib.util.find_spec("av")))'
  fi
done

for root in \
  "/home/benjamin/.cache/huggingface" \
  "/home/benjamin/.cache/lerobot" \
  "/mnt/c/Users/benja/.cache/huggingface" \
  "$repo_root/datasets" \
  "$repo_root/questions/results"
do
  if [[ -e "$root" ]]; then
    echo "asset_root=$root"
    find "$root" -maxdepth 5 -type d \( -iname '*smolvla*' -o -iname '*lerobot--pusht*' -o -name pretrained_model \) -print | head -40
  fi
done

echo "smolvla_local_probe_done"
