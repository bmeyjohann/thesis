#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-/home/benjamin/thesis}"
LOG_DIR="${UNITREE_WSLG_LOG_DIR:-$ROOT_DIR/logs/wslg}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/preflight_$STAMP.log"

{
  echo "=== Unitree WSLg preflight $(date --iso-8601=seconds) ==="
  echo "kernel: $(uname -r)"
  if [[ -r /mnt/wslg/versions.txt ]]; then
    echo "=== WSLg versions ==="
    cat /mnt/wslg/versions.txt
  fi
  echo "=== active Unitree/mjlab processes ==="
  STALE_PROCESSES="$(
    ps -eo pid,ppid,stat,etime,cmd |
      grep -E 'eval_interactive_unitree_nav|train_unitree_nav|python.*mjlab' |
      grep -v -E 'grep|unitree_wslg_preflight' || true
  )"
  if [[ -n "$STALE_PROCESSES" ]]; then
    printf '%s\n' "$STALE_PROCESSES"
    echo "WARNING: an existing Unitree/mjlab process may own a WSLg window."
    if [[ "${UNITREE_FAIL_ON_STALE_VIEWER:-0}" == "1" ]]; then
      echo "Refusing to launch because UNITREE_FAIL_ON_STALE_VIEWER=1."
      exit 3
    fi
  else
    echo "<none>"
  fi
  echo "=== recent WSLg transport/error lines ==="
  if [[ -r /mnt/wslg/weston.log ]]; then
    grep -Eai 'vail|rail|copy mode|shared.?memory|virtio|error|fatal|disconnect' /mnt/wslg/weston.log |
      tail -150 || true
  else
    echo "<weston.log unavailable>"
  fi
} | tee "$LOG_PATH"

if grep -Eaiq 'rdp_allocate_shared_memory: Failed|vail.*fail|warn: copy mode' "$LOG_PATH"; then
  echo "[wslg-preflight] WARNING: WSLg shared-memory transport failed; the viewer may open in COPY MODE." >&2
  echo "[wslg-preflight] Recover Windows-side msrdc/Explorer or terminate Ubuntu before blaming the Unitree renderer." >&2
fi
echo "[wslg-preflight] diagnostic: $LOG_PATH"
