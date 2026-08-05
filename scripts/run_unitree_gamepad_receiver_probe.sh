#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

exec "$PYTHON_BIN" "$ROOT_DIR/tools/probe_unitree_gamepad_receiver.py" "$@"

