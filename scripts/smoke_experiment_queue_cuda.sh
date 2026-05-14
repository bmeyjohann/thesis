#!/usr/bin/env bash
set -euo pipefail

echo "smoke queue cuda: start"
echo "cwd=$(pwd)"
echo "hostname=$(hostname)"
echo "python=$(command -v python || true)"
echo "conda_env=${CONDA_DEFAULT_ENV:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi: begin"
  nvidia-smi
  echo "nvidia-smi: end"
else
  echo "nvidia-smi: missing"
  exit 10
fi

python - <<'PY'
import torch

print("torch", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("device_count", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA not available in queue worker environment")

print("device0", torch.cuda.get_device_name(0), flush=True)
x = torch.tensor([1.0], device="cuda:0")
print("alloc_ok", x.item(), flush=True)
PY

echo "smoke queue cuda: done"
