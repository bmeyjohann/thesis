#!/bin/bash
# Smart SLURM job submission with automatic account detection
# Usage: bash submit_job.sh [script.sbatch]

SCRIPT_NAME="${1:-run_experiment.sbatch}"

echo "🚀 Smart SLURM Job Submission"
echo "Script: $SCRIPT_NAME"

# Function to detect user's SLURM account
detect_account() {
    echo "🔍 Auto-detecting SLURM account..." >&2
    
    # Method 1: Check saved config file
    if [[ -f ".slurm_account" ]]; then
        source .slurm_account
        if [[ -n "$SLURM_ACCOUNT" ]]; then
            echo "✓ Found account in .slurm_account: $SLURM_ACCOUNT" >&2
            echo "$SLURM_ACCOUNT"
            return 0
        fi
    fi
    
    # Method 2: Check environment variable
    if [[ -n "$SLURM_ACCOUNT" ]]; then
        echo "✓ Found account in SLURM_ACCOUNT: $SLURM_ACCOUNT" >&2
        echo "$SLURM_ACCOUNT"
        return 0
    fi
    
    # Method 3: Check sacctmgr (most reliable)
    if command -v sacctmgr >/dev/null 2>&1; then
        # Get user's available accounts
        ACCOUNTS=$(sacctmgr show assoc user=$USER -P -n | cut -d'|' -f2 | sort -u | grep -v '^$')
        
        if [[ -n "$ACCOUNTS" ]]; then
            ACCOUNT_COUNT=$(echo "$ACCOUNTS" | wc -l)
            
            if [[ $ACCOUNT_COUNT -eq 1 ]]; then
                echo "✓ Found single account: $ACCOUNTS" >&2
                echo "$ACCOUNTS"
                return 0
            else
                echo "⚠️  Multiple accounts found:" >&2
                echo "$ACCOUNTS" | nl >&2
                echo "" >&2
                echo "Please set SLURM_ACCOUNT environment variable:" >&2
                echo "  export SLURM_ACCOUNT=your_account_name" >&2
                echo "  bash submit_job.sh" >&2
                exit 1
            fi
        fi
    fi
    
    # Method 4: Check sshare (fallback)
    if command -v sshare >/dev/null 2>&1; then
        ACCOUNT=$(sshare -U | grep "^$USER" | awk '{print $2}' | head -1)
        if [[ -n "$ACCOUNT" && "$ACCOUNT" != "Account" ]]; then
            echo "✓ Found account via sshare: $ACCOUNT" >&2
            echo "$ACCOUNT"
            return 0
        fi
    fi
    
    # Method 5: Interactive prompt
    echo "❌ Could not auto-detect account." >&2
    echo -n "Please enter your SLURM account name: " >&2
    read -r ACCOUNT
    if [[ -n "$ACCOUNT" ]]; then
        echo "$ACCOUNT"
        return 0
    fi
    
    echo "❌ No account provided. Exiting." >&2
    exit 1
}

# Check if script exists
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "❌ Script not found: $SCRIPT_NAME"
    echo "Available .sbatch files:"
    ls -1 *.sbatch 2>/dev/null || echo "  No .sbatch files found"
    exit 1
fi

# Detect account
USER_ACCOUNT=$(detect_account)

echo "👤 Using account: $USER_ACCOUNT"

# Create temporary script with account replaced
TEMP_SCRIPT="$(mktemp --suffix=.sbatch)"
echo "🔧 Replacing <your_account> with: $USER_ACCOUNT"

# Simple replacement using awk (most reliable)
awk -v account="$USER_ACCOUNT" '{gsub(/<your_account>/, account); print}' "$SCRIPT_NAME" > "$TEMP_SCRIPT"

echo "📝 Generated temporary script: $TEMP_SCRIPT"

# Verify the script is not empty and replacement worked
if [[ ! -s "$TEMP_SCRIPT" ]]; then
    echo "❌ Generated script is empty!"
    echo "Original script exists: $(ls -la "$SCRIPT_NAME")"
    exit 1
fi

# Show the account line to confirm
echo "📋 Account configuration:"
ACCOUNT_LINE=$(grep "^#SBATCH --account=" "$TEMP_SCRIPT")
if [[ -n "$ACCOUNT_LINE" ]]; then
    echo "   $ACCOUNT_LINE"
else
    echo "❌ No account line found in generated script!"
    echo "Script contents:"
    head -10 "$TEMP_SCRIPT"
    exit 1
fi

# Ask for confirmation
echo ""
read -p "🤔 Submit job with this configuration? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Submitting job..."
    
    # Submit the job
    JOB_OUTPUT=$(sbatch "$TEMP_SCRIPT")
    
    if [[ $? -eq 0 ]]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+')
        echo "✅ Job submitted successfully!"
        echo "   Job ID: $JOB_ID"
        echo "   Account: $USER_ACCOUNT"
        echo "   Script: $SCRIPT_NAME"
        echo ""
        echo "📊 Monitor with:"
        echo "   squeue -j $JOB_ID"
        echo "   tail -f logs/reward_comparison_${JOB_ID}.out"
        
        # Save account for future use
        echo "export SLURM_ACCOUNT=\"$USER_ACCOUNT\"" > .slurm_account
        echo "💾 Saved account to .slurm_account for future use"
        
        # Start background monitoring for periodic WandB sync
        if [[ -n "$JOB_ID" ]]; then
            echo ""
            echo "🔧 Background Monitoring Options:"
            echo "   1. Start background monitor with periodic WandB sync (recommended)"
            echo "   2. Skip background monitoring (manual sync later)"
            echo ""
            read -p "Start background monitoring? (Y/n): " -n 1 -r
            echo
            
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                # Ask for sync interval
                echo ""
                read -p "WandB sync interval in minutes (default 30): " SYNC_INTERVAL
                SYNC_INTERVAL=${SYNC_INTERVAL:-30}
                
                # Make background monitor executable
                chmod +x background_monitor.sh
                
                # Start background monitor with nohup
                echo "🚀 Starting background monitor for job $JOB_ID (sync every $SYNC_INTERVAL minutes)..."
                nohup bash background_monitor.sh "$JOB_ID" "$SYNC_INTERVAL" >/dev/null 2>&1 &
                MONITOR_PID=$!
                
                echo "✅ Background monitor started (PID: $MONITOR_PID)"
                echo "📝 Monitor log: logs/background_monitor_${JOB_ID}.log"
                echo "🔍 Check status with: tail -f logs/background_monitor_${JOB_ID}.log"
                echo ""
                echo "💡 The terminal can now be closed safely!"
                echo "   Your job will continue running and WandB will sync automatically."
                echo ""
                echo "🛠️  Manual commands (if needed):"
                echo "   Monitor job: squeue -j $JOB_ID"
                echo "   Job logs: tail -f logs/ts_sac_batchC_term_${JOB_ID}.out"
                echo "   Stop monitor: pkill -f \"background_monitor.sh $JOB_ID\""
                echo "   Manual sync: bash -c 'source sc_venv_template/activate.sh && python -m wandb sync wandb/offline*'"
            else
                echo "⏭️  Skipping background monitoring"
                echo "📝 To manually sync WandB later, run:"
                echo "   bash background_monitor.sh $JOB_ID"
            fi
        fi
    else
        echo "❌ Job submission failed!"
        echo "Error output:"
        echo "$JOB_OUTPUT"
    fi
else
    echo "❌ Job submission cancelled"
fi

# Clean up temporary file
rm -f "$TEMP_SCRIPT"
echo "🧹 Cleaned up temporary files"
