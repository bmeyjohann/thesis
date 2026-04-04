#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

exec /usr/bin/python3 "${REPO_ROOT}/tools/experiment_queue_mcp/daemon.py" \
  --queue-root "${REPO_ROOT}/experiment_queue" \
  --workspace-root "${REPO_ROOT}" \
  --default-cwd "${REPO_ROOT}" \
  --script-root "${REPO_ROOT}/scripts" \
  --script-root "${REPO_ROOT}" \
  --default-conda-env fasttd3 \
  --conda-sh-path "/home/benjamin/miniconda3/etc/profile.d/conda.sh" \
  --poll-interval 1.0 \
  --terminate-grace 10.0 \
  --max-concurrent-jobs 2
