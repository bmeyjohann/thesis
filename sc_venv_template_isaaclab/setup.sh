#!/bin/bash
set -euo pipefail

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}/config.sh"

PYTHON_BIN="${PYTHON_BIN:-/isaac-sim/python.sh}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Warning: ${PYTHON_BIN} not found, falling back to python3 on PATH."
    PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Unable to locate a Python interpreter. Set PYTHON_BIN=/path/to/python before running setup."
    exit 1
fi

if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Creating virtual environment at ${ENV_DIR} using ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m venv --prompt "${ENV_NAME}" --system-site-packages "${ENV_DIR}"
else
    echo "Using existing virtual environment at ${ENV_DIR}."
fi

VENV_PYTHON="${ENV_DIR}/bin/python"

echo "Upgrading pip tooling inside the virtual environment..."
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "Installing project Python requirements..."
"${VENV_PYTHON}" -m pip install --upgrade --requirement "${ABSOLUTE_PATH}/requirements.txt"

REPO_ROOT="$(realpath "${ABSOLUTE_PATH}/..")"

ISAACLAB_PATH="${ISAACLAB_PATH:-${REPO_ROOT}/IsaacLab}"
if [[ -d "${ISAACLAB_PATH}" ]]; then
    echo "Installing IsaacLab from ${ISAACLAB_PATH} (editable, no dependencies)..."
    "${VENV_PYTHON}" -m pip install --no-deps -e "${ISAACLAB_PATH}"
else
    echo "IsaacLab source tree not found at ${ISAACLAB_PATH}; skipping editable install."
fi

FASTTD3_PATH="${FASTTD3_PATH:-${REPO_ROOT}/fasttd3}"
if [[ -d "${FASTTD3_PATH}" ]]; then
    echo "Installing fasttd3 from ${FASTTD3_PATH} (editable, no dependencies)..."
    "${VENV_PYTHON}" -m pip install --no-deps -e "${FASTTD3_PATH}"
else
    echo "fasttd3 source tree not found at ${FASTTD3_PATH}; skipping editable install."
fi

echo "Setup complete. Activate the environment with: source ${ABSOLUTE_PATH}/activate.sh"
