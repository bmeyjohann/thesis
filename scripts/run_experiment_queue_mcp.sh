#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
queue_root="${EXPERIMENT_QUEUE_ROOT:-$repo_root/experiment_queue}"
workspace_root="${EXPERIMENT_QUEUE_WORKSPACE:-$repo_root}"
default_cwd="${EXPERIMENT_QUEUE_CWD:-$repo_root}"
default_conda_env="${EXPERIMENT_QUEUE_CONDA_ENV:-fasttd3}"
conda_sh_path="${EXPERIMENT_QUEUE_CONDA_SH:-/home/benjamin/miniconda3/etc/profile.d/conda.sh}"
poll_interval="${EXPERIMENT_QUEUE_POLL_INTERVAL:-1.0}"
terminate_grace="${EXPERIMENT_QUEUE_TERMINATE_GRACE:-10.0}"

exec python3 "$repo_root/tools/experiment_queue_mcp/server.py" \
  --queue-root "$queue_root" \
  --workspace-root "$workspace_root" \
  --default-cwd "$default_cwd" \
  --script-root "$repo_root/scripts" \
  --script-root "$repo_root" \
  --default-conda-env "$default_conda_env" \
  --conda-sh-path "$conda_sh_path" \
  --poll-interval "$poll_interval" \
  --terminate-grace "$terminate_grace"

