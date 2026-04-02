#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${UVX_BIN:-}" ]]; then
  uvx_bin="${UVX_BIN}"
elif command -v uvx >/dev/null 2>&1; then
  uvx_bin="$(command -v uvx)"
elif [[ -x "${HOME}/.local/bin/uvx" ]]; then
  uvx_bin="${HOME}/.local/bin/uvx"
else
  echo "uvx not found. Install uv or set UVX_BIN before starting Codex." >&2
  exit 1
fi

exec "${uvx_bin}" --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
