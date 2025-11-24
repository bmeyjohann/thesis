#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--sync-wandb] [--wandb-dir DIR] [--wandb-interval SEC] [sbatch_file]
Defaults to sbatch_file=run_experiment.sbatch when omitted.
EOF
}

SYNC_WANDB=0
WAND_DIR="wandb"
WAND_INTERVAL=600
SCRIPT_NAME=""
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync-wandb)
            SYNC_WANDB=1
            ;;
        --wandb-dir)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --wandb-dir" >&2; exit 1; }
            WAND_DIR="$1"
            ;;
        --wandb-interval)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --wandb-interval" >&2; exit 1; }
            WAND_INTERVAL="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            SCRIPT_NAME="$1"
            ;;
    esac
    shift || true
done

SCRIPT_NAME="${SCRIPT_NAME:-run_experiment.sbatch}"

detect_account() {
    if [[ -f .slurm_account ]]; then
        # shellcheck source=/dev/null
        source .slurm_account
        [[ -n "${SLURM_ACCOUNT:-}" ]] && { echo "$SLURM_ACCOUNT"; return 0; }
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        echo "$SLURM_ACCOUNT"
        return 0
    fi
    if command -v sacctmgr >/dev/null 2>&1; then
        local accounts
        accounts=$(sacctmgr show assoc user="$USER" -P -n | cut -d'|' -f2 | sort -u | grep -v '^$' || true)
        if [[ -n "$accounts" ]]; then
            if [[ $(echo "$accounts" | wc -l) -eq 1 ]]; then
                echo "$accounts"
                return 0
            fi
            echo "Multiple SLURM accounts detected:" >&2
            echo "$accounts" | nl >&2
            echo "Set SLURM_ACCOUNT and re-run." >&2
            exit 1
        fi
    fi
    read -rp "Enter SLURM account: " manual
    [[ -n "$manual" ]] || { echo "No account provided." >&2; exit 1; }
    echo "$manual"
}

monitor_wandb() {
    local job_id="$1"
    local wandb_dir="$2"
    local interval="$3"
    source_sc_env || true
    if ! command -v wandb >/dev/null 2>&1; then
        echo "⚠️  wandb CLI not found; skipping auto-sync."
        return
    fi
    mkdir -p "$wandb_dir"
    echo "🔁 Auto-syncing wandb runs from ${wandb_dir} every ${interval}s while job ${job_id} is active."
    while squeue -h -j "$job_id" >/dev/null 2>&1; do
        wandb sync --sync-all "$wandb_dir" >/dev/null 2>&1 || true
        sleep "$interval"
    done
    wandb sync --sync-all "$wandb_dir" >/dev/null 2>&1 || true
    echo "✅ wandb sync finished for job ${job_id}."
}

if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "❌ Script not found: $SCRIPT_NAME"
    ls -1 *.sbatch 2>/dev/null || true
    exit 1
fi

ACCOUNT=$(detect_account)
echo "Submitting $SCRIPT_NAME with account $ACCOUNT"

TEMP_SCRIPT=$(mktemp --suffix=.sbatch)
trap 'rm -f "$TEMP_SCRIPT"' EXIT
awk -v account="$ACCOUNT" '{gsub(/<your_account>/, account); print}' "$SCRIPT_NAME" >"$TEMP_SCRIPT"

if [[ ! -s "$TEMP_SCRIPT" ]]; then
    echo "❌ Failed to generate temporary script"
    exit 1
fi

if ! grep -q "^#SBATCH --account=" "$TEMP_SCRIPT"; then
    echo "❌ Generated script is missing an --account directive"
    exit 1
fi

JOB_OUTPUT=$(sbatch "$TEMP_SCRIPT" 2>&1) || {
    echo "❌ sbatch failed"
    echo "$JOB_OUTPUT"
    exit 1
}

JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+' | tail -n1)
echo "✅ Submitted: $JOB_OUTPUT"
echo "Use: squeue -j $JOB_ID"

printf 'export SLURM_ACCOUNT="%s"\n' "$ACCOUNT" > .slurm_account

if [[ "$SYNC_WANDB" -eq 1 ]]; then
    mkdir -p logs
    LOG_PATH="logs/wandb_sync_${JOB_ID}.log"
    echo "📡 Launching background wandb sync (log: ${LOG_PATH})"
    (
        source_sc_env || true
        monitor_wandb "$JOB_ID" "$WAND_DIR" "$WAND_INTERVAL"
    ) >"${LOG_PATH}" 2>&1 < /dev/null &
    disown
fi
