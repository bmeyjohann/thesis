#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0
Interactive wandb sync for offline runs modified within the last N days.
EOF
}

WAND_DIR="wandb"
DAYS=1
SC_ACTIVATE_SCRIPT="${SC_ACTIVATE_SCRIPT:-sc_venv_template/activate.sh}"
_SC_ENV_SOURCED=0

source_sc_env() {
    if [[ "$_SC_ENV_SOURCED" -eq 1 ]]; then
        return 0
    fi
    if [[ -f "$SC_ACTIVATE_SCRIPT" ]]; then
        echo "🔧 Sourcing ${SC_ACTIVATE_SCRIPT} for wandb."
        # shellcheck source=/dev/null
        source "$SC_ACTIVATE_SCRIPT"
        _SC_ENV_SOURCED=1
        return 0
    fi
    echo "⚠️  Missing activate script (${SC_ACTIVATE_SCRIPT}); wandb sync may fail." >&2
    return 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

read -rp "Enable wandb logging sync? [Y/n] " enable_sync
case "${enable_sync:-Y}" in
    Y|y|"")
        ;;
    N|n)
        echo "Skipping wandb sync."
        exit 0
        ;;
    *)
        echo "Invalid response; defaulting to yes."
        ;;
esac

read -rp "Base wandb directory [${WAND_DIR}]: " input_wand_dir
WAND_DIR="${input_wand_dir:-$WAND_DIR}"

read -rp "Sync runs modified in the last N days [${DAYS}]: " input_days
DAYS="${input_days:-$DAYS}"

if ! [[ "$DAYS" =~ ^[0-9]+$ ]]; then
    echo "❌ --days must be a non-negative integer." >&2
    exit 1
fi

source_sc_env || true

if ! command -v wandb >/dev/null 2>&1; then
    echo "❌ wandb CLI not found; activate your environment first." >&2
    exit 1
fi

if [[ ! -d "$WAND_DIR" ]]; then
    echo "❌ wandb directory not found: $WAND_DIR" >&2
    exit 1
fi

mapfile -d '' wandb_dirs < <(find "$WAND_DIR" -type d -name "wandb" -print0)
if [[ "${#wandb_dirs[@]}" -eq 0 ]]; then
    wandb_dirs=("$WAND_DIR")
fi

run_dirs=()
for root in "${wandb_dirs[@]}"; do
    while IFS= read -r -d '' run_dir; do
        run_dirs+=("$run_dir")
    done < <(find "$root" -type d \( -name "run-*" -o -name "offline-run-*" \) -mtime "-${DAYS}" -print0)
done

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
    echo "ℹ️  No wandb runs modified in the last ${DAYS} day(s) under ${WAND_DIR}."
    exit 0
fi

echo "📡 Syncing ${#run_dirs[@]} wandb run(s) from the last ${DAYS} day(s)..."
for run_dir in "${run_dirs[@]}"; do
    echo "  - $run_dir"
    WANDB_MODE=online wandb sync --sync-all "$run_dir" >/dev/null 2>&1 || true
done
echo "✅ wandb sync complete."
