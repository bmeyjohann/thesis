#!/usr/bin/env bash
set -euo pipefail

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/benjamin/miniconda3/envs/fasttd3/bin/python" ]]; then
    PYTHON_BIN="/home/benjamin/miniconda3/envs/fasttd3/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No usable python interpreter found." >&2
    exit 127
  fi
fi

echo "CUDA queue smoke using: ${PYTHON_BIN}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

"${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(3)

name = torch.cuda.get_device_name(0)
print(f"device_name={name}")

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
print(f"cuda_tensor={x.tolist()}")
print("cuda_smoke=ok")
PY
