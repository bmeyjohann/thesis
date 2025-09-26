#!/bin/bash
# Background job monitor with periodic WandB syncing
# Usage: bash background_monitor.sh JOB_ID [sync_interval_minutes]

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 JOB_ID [sync_interval_minutes]"
    echo "Example: $0 12345 30"
    exit 1
fi

JOB_ID="$1"
SYNC_INTERVAL="${2:-30}"  # Default: 30 minutes
SYNC_INTERVAL_SECONDS=$((SYNC_INTERVAL * 60))

LOG_FILE="logs/background_monitor_${JOB_ID}.log"
mkdir -p logs

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to sync WandB
sync_wandb() {
    log "🔄 Starting WandB sync attempt..."
    
    if [[ -f sc_venv_template/activate.sh ]]; then
        (
            set +e
            source sc_venv_template/activate.sh
            if python -c 'import wandb; print("wandb-ok")' >/dev/null 2>&1; then
                log "☁️  Syncing wandb offline runs..."
                
                # Count runs before sync
                OFFLINE_RUNS=$(find wandb -name "offline-run-*" -type d 2>/dev/null | wc -l)
                OFFLINE_RUNS_NESTED=$(find wandb -path "*/offline*/*" -type d 2>/dev/null | wc -l)
                
                if [[ $OFFLINE_RUNS -gt 0 ]] || [[ $OFFLINE_RUNS_NESTED -gt 0 ]]; then
                    log "📊 Found $OFFLINE_RUNS top-level and $OFFLINE_RUNS_NESTED nested offline runs"
                    
                    # Sync with timeout to prevent hanging
                    timeout 300 bash -c '
                        WANDB_MODE=run WANDB_CONSOLE=off WANDB_SILENT=true \
                          python -m wandb sync wandb/offline-run-* 2>/dev/null || true
                        WANDB_MODE=run WANDB_CONSOLE=off WANDB_SILENT=true \
                          python -m wandb sync wandb/offline*/* 2>/dev/null || true
                    '
                    
                    if [[ $? -eq 0 ]]; then
                        log "✅ WandB sync completed successfully"
                    elif [[ $? -eq 124 ]]; then
                        log "⏰ WandB sync timed out after 5 minutes"
                    else
                        log "⚠️  WandB sync completed with warnings"
                    fi
                else
                    log "📭 No offline runs found to sync"
                fi
            else
                log "⚠️  wandb not available in environment; skipping sync"
            fi
        )
    else
        log "⚠️  sc_venv_template/activate.sh not found; skipping WandB sync"
    fi
}

# Function to check if job is still running
is_job_running() {
    squeue -h -j "$JOB_ID" >/dev/null 2>&1
}

# Function to get job status
get_job_status() {
    if is_job_running; then
        echo "RUNNING"
    else
        # Check if job completed successfully or failed
        sacct -j "$JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' '
    fi
}

log "🚀 Starting background monitor for job $JOB_ID"
log "📅 Sync interval: $SYNC_INTERVAL minutes"
log "📝 Monitor log: $LOG_FILE"
log "🔍 Job status check command: squeue -j $JOB_ID"

# Initial status check
if ! is_job_running; then
    log "❌ Job $JOB_ID is not running or does not exist"
    exit 1
fi

log "✅ Job $JOB_ID is running, starting monitoring..."

# Main monitoring loop
LAST_SYNC=0
while true; do
    STATUS=$(get_job_status)
    CURRENT_TIME=$(date +%s)
    
    if [[ "$STATUS" != "RUNNING" ]]; then
        log "🏁 Job $JOB_ID finished with status: $STATUS"
        log "🔄 Performing final WandB sync..."
        sync_wandb
        log "✅ Background monitoring completed for job $JOB_ID"
        break
    fi
    
    # Check if it's time for periodic sync
    TIME_SINCE_LAST_SYNC=$((CURRENT_TIME - LAST_SYNC))
    if [[ $TIME_SINCE_LAST_SYNC -ge $SYNC_INTERVAL_SECONDS ]]; then
        log "⏰ Periodic sync time (every $SYNC_INTERVAL minutes)"
        sync_wandb
        LAST_SYNC=$CURRENT_TIME
    fi
    
    # Log status every hour
    HOUR_SECONDS=3600
    if [[ $((CURRENT_TIME % HOUR_SECONDS)) -lt 60 ]]; then
        log "📊 Job $JOB_ID still running (periodic status update)"
    fi
    
    # Wait 60 seconds before next check
    sleep 60
done

log "🧹 Background monitor for job $JOB_ID completed"
