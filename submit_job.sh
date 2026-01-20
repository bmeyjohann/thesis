#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [sbatch_file]
Defaults to sbatch_file=run_experiment.sbatch when omitted.
EOF
}

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

collect_run_dirs() {
    local base_dir="$1"
    local -a wandb_dirs=()
    local -a run_dirs=()
    while IFS= read -r -d '' wandb_dir; do
        wandb_dirs+=("$wandb_dir")
    done < <(find "$base_dir" -type d -name "wandb" -print0)
    if [[ "${#wandb_dirs[@]}" -eq 0 ]]; then
        wandb_dirs=("$base_dir")
    fi
    for root in "${wandb_dirs[@]}"; do
        while IFS= read -r -d '' run_dir; do
            run_dirs+=("$run_dir")
        done < <(find "$root" -type d \( -name "run-*" -o -name "offline-run-*" \) -print0)
    done
    printf '%s\n' "${run_dirs[@]}"
}

monitor_wandb() {
    local job_id="$1"
    local wandb_dir="$2"
    local interval="$3"
    local baseline_file="$4"
    source_sc_env || true
    if ! command -v wandb >/dev/null 2>&1; then
        echo "⚠️  wandb CLI not found; skipping auto-sync."
        return
    fi
    mkdir -p "$wandb_dir"
    echo "🔁 Auto-syncing new wandb runs from ${wandb_dir} every ${interval}s while job ${job_id} is active."

    declare -A baseline_dirs
    if [[ -f "$baseline_file" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && baseline_dirs["$line"]=1
        done < "$baseline_file"
    fi

    declare -A new_run_dirs
    while true; do
        local queue_output
        queue_output=$(squeue -h -j "$job_id" -o "%i" 2>/dev/null || true)
        [[ -n "$queue_output" ]] || break

        while IFS= read -r run_dir; do
            [[ -n "$run_dir" ]] || continue
            if [[ -z "${baseline_dirs[$run_dir]:-}" ]]; then
                new_run_dirs["$run_dir"]=1
            fi
        done < <(collect_run_dirs "$wandb_dir")

        for run_dir in "${!new_run_dirs[@]}"; do
            WANDB_MODE=online wandb sync --sync-all "$run_dir" >/dev/null 2>&1 || true
        done
        sleep "$interval"
    done

    for run_dir in "${!new_run_dirs[@]}"; do
        WANDB_MODE=online wandb sync --sync-all "$run_dir" >/dev/null 2>&1 || true
    done
    echo "✅ wandb sync finished for job ${job_id}."
}

if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "❌ Script not found: $SCRIPT_NAME"
    ls -1 *.sbatch 2>/dev/null || true
    exit 1
fi

SYNC_WANDB=1
WAND_DIR="wandb"
WAND_INTERVAL=600
BASELINE_FILE=""

read -rp "Enable wandb logging sync? [Y/n] " enable_sync
case "${enable_sync:-Y}" in
    Y|y|"")
        ;;
    N|n)
        SYNC_WANDB=0
        ;;
    *)
        echo "Invalid response; defaulting to yes."
        ;;
esac

if [[ "$SYNC_WANDB" -eq 1 ]]; then
    read -rp "Base wandb directory [${WAND_DIR}]: " input_wand_dir
    WAND_DIR="${input_wand_dir:-$WAND_DIR}"

    read -rp "Sync interval seconds [${WAND_INTERVAL}]: " input_interval
    WAND_INTERVAL="${input_interval:-$WAND_INTERVAL}"

    if ! [[ "$WAND_INTERVAL" =~ ^[0-9]+$ ]]; then
        echo "❌ Sync interval must be a non-negative integer." >&2
        exit 1
    fi

    mkdir -p logs
    BASELINE_FILE=$(mktemp)
    collect_run_dirs "$WAND_DIR" > "$BASELINE_FILE" || true
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
    BASELINE_PATH="logs/wandb_sync_${JOB_ID}_baseline.txt"
    if [[ -n "${BASELINE_FILE:-}" && -f "$BASELINE_FILE" ]]; then
        mv "$BASELINE_FILE" "$BASELINE_PATH"
    else
        : > "$BASELINE_PATH"
    fi
    echo "📡 Launching background wandb sync (log: ${LOG_PATH})"
    (
        source_sc_env || true
        monitor_wandb "$JOB_ID" "$WAND_DIR" "$WAND_INTERVAL" "$BASELINE_PATH"
    ) >"${LOG_PATH}" 2>&1 < /dev/null &
    disown
fi
