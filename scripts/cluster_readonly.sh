#!/usr/bin/env bash
set -euo pipefail

# Read-only helper for cluster inspection.
# Restricts commands to an allowlist and blocks shell metacharacters that could
# be used for write/exec chaining.

HOST="${CLUSTER_HOST:-juwels-booster.fz-juelich.de}"
REMOTE_ROOT="${CLUSTER_ROOT:-/p/project1/hai_1074/meyjohann1/thesis}"

usage() {
  cat <<'EOF'
Usage:
  scripts/cluster_readonly.sh <allowed_cmd> [args...]

Allowed commands:
  ls cat tail head grep find wc stat du df pwd basename dirname sed

Environment overrides:
  CLUSTER_HOST, CLUSTER_ROOT

Examples:
  scripts/cluster_readonly.sh ls -1 logs
  scripts/cluster_readonly.sh tail -n 120 logs/drqv2_pvp_storage_debug_13223266.out
  scripts/cluster_readonly.sh find wandb -maxdepth 2 -type d -name 'offline-run-*'
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

cmd="$1"
shift

allowed=(
  ls
  cat
  tail
  head
  grep
  find
  wc
  stat
  du
  df
  pwd
  basename
  dirname
  sed
)

is_allowed=0
for c in "${allowed[@]}"; do
  if [[ "$cmd" == "$c" ]]; then
    is_allowed=1
    break
  fi
done

if [[ "$is_allowed" -ne 1 ]]; then
  echo "Denied: command '$cmd' is not in allowlist." >&2
  usage
  exit 2
fi

# Deny shell control characters in any argument.
for token in "$cmd" "$@"; do
  if [[ "$token" =~ [\;\&\|\>\<\`\$\(\)] ]]; then
    echo "Denied: token '$token' contains blocked shell metacharacters." >&2
    exit 3
  fi
done

escaped_root="$(printf '%q' "$REMOTE_ROOT")"
escaped_cmd="$(printf '%q' "$cmd")"
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    escaped_cmd+=" $(printf '%q' "$arg")"
  done
fi

remote_line="cd ${escaped_root} && ${escaped_cmd}"
exec ssh "$HOST" "$remote_line"
