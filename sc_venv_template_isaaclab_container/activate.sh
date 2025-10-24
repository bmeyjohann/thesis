#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

if [[ "$0" == "${SOURCE_PATH}" ]]; then
    echo "This activation script must be sourced, e.g. 'source ${SOURCE_PATH}'."
    exit 1
fi

source "${ABSOLUTE_PATH}/config.sh"

if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
    echo "Virtual environment not found at ${ENV_DIR}. Run setup.sh first."
    return 1
fi

source "${ENV_DIR}/bin/activate"

# Discover the venv's site-packages path (python version may vary)
_PY_DIR="$(find "${ENV_DIR}/lib" -maxdepth 1 -type d -name "python*" | head -n 1)"
if [[ -z "${_PY_DIR}" ]]; then
    echo "Unable to locate site-packages under ${ENV_DIR}. Has the venv been created?"
    return 1
fi
_VENV_SITE_PACKAGES="${_PY_DIR}/site-packages"

# Extend PYTHONPATH so IsaacLab sources and venv packages are discoverable.
_PYTHONPATH_ENTRIES="${_VENV_SITE_PACKAGES}"
if [[ -d /workspace/IsaacLab/source ]]; then
    _PYTHONPATH_ENTRIES="/workspace/IsaacLab/source:${_PYTHONPATH_ENTRIES}"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${_PYTHONPATH_ENTRIES}:${PYTHONPATH}"
else
    export PYTHONPATH="${_PYTHONPATH_ENTRIES}"
fi

unset _PY_DIR _VENV_SITE_PACKAGES _PYTHONPATH_ENTRIES
