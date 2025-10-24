#!/bin/bash

# See https://stackoverflow.com/a/28336473
SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "The activation script must be sourced, otherwise the virtual environment will not work." || ( echo "Vars script must be sourced." && exit 1) ;

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

export PYTHONPATH="$(echo "${ENV_DIR}"/lib/python*/site-packages):${PYTHONPATH}"

export ISAACSIM_PATH="${HOME}/${USER}/isaacsim5.1.0"
export ISAACSIM_PYTHON_PATH="${ISAACSIM_PATH}/python.sh"

source "${ENV_DIR}"/bin/activate
