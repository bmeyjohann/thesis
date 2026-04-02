#!/usr/bin/env bash
set -euo pipefail

export ZOTERO_LOCAL="${ZOTERO_LOCAL:-true}"

if command -v powershell.exe >/dev/null 2>&1; then
  exec powershell.exe -NoProfile -Command "uvx --from git+https://github.com/54yyyu/zotero-mcp.git zotero-mcp"
fi

if [[ -n "${UVX_BIN:-}" ]]; then
  uvx_bin="${UVX_BIN}"
elif command -v uvx >/dev/null 2>&1; then
  uvx_bin="$(command -v uvx)"
elif [[ -x "${HOME}/.local/bin/uvx" ]]; then
  uvx_bin="${HOME}/.local/bin/uvx"
else
  echo "uvx not found and powershell.exe is unavailable. Install uv or set UVX_BIN." >&2
  exit 1
fi

exec "${uvx_bin}" --from git+https://github.com/54yyyu/zotero-mcp.git zotero-mcp
