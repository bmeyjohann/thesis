#!/bin/bash
SCRIPT_NAME="${1:-run_experiment.sbatch}"

echo "🚀 SLURM submission helper"
echo "Script: $SCRIPT_NAME"

detect_account() {
    if [[ -f .slurm_account ]]; then
        source .slurm_account
        [[ -n "$SLURM_ACCOUNT" ]] && echo "$SLURM_ACCOUNT" && return 0
    fi
    if [[ -n "$SLURM_ACCOUNT" ]];then
        echo "$SLURM_ACCOUNT" && return 0
    fi
    if command -v sacctmgr >/dev/null 2>&1; then
        local accounts
        accounts=$(sacctmgr show assoc user="$USER" -P -n | cut -d'|' -f2 | sort -u | grep -v '^$')
        if [[ -n "$accounts" ]]; then
            if [[ $(echo "$accounts" | wc -l) -eq 1 ]]; then
                echo "$accounts"
                return 0
            fi
            echo "Multiple accounts detected:" >&2
            echo "$accounts" | nl >&2
            echo "Set SLURM_ACCOUNT and re-run." >&2
            exit 1
        fi
    fi
    echo -n "Enter SLURM account: " >&2
    read -r manual
    [[ -n "$manual" ]] || { echo "No account provided." >&2; exit 1; }
    echo "$manual"
}

if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "❌ Script not found: $SCRIPT_NAME"
    ls -1 *.sbatch 2>/dev/null && exit 1
    exit 1
fi

ACCOUNT=$(detect_account)
echo "👤 Using account: $ACCOUNT"

TEMP_SCRIPT=$(mktemp --suffix=.sbatch)
trap 'rm -f "$TEMP_SCRIPT"' EXIT
awk -v account="$ACCOUNT" '{gsub(/<your_account>/, account); print}' "$SCRIPT_NAME" >"$TEMP_SCRIPT"

if [[ ! -s "$TEMP_SCRIPT" ]]; then
    echo "❌ Failed to generate temporary script"
    exit 1
fi

ACCOUNT_LINE=$(grep "^#SBATCH --account=" "$TEMP_SCRIPT")
if [[ -z "$ACCOUNT_LINE" ]]; then
    echo "❌ Generated script is missing an --account directive"
    exit 1
fi
echo "📋 $ACCOUNT_LINE"

read -p "Submit this job? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

JOB_OUTPUT=$(sbatch "$TEMP_SCRIPT" 2>&1)
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
    echo "❌ sbatch failed"
    echo "$JOB_OUTPUT"
    exit $STATUS
fi

JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+')
echo "✅ Submitted: $JOB_OUTPUT"
echo "   Monitor with: squeue -j $JOB_ID"

echo "export SLURM_ACCOUNT=\"$ACCOUNT\"" > .slurm_account
echo "💾 Saved account to .slurm_account"
