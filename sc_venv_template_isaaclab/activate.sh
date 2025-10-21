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
