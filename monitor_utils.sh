#!/bin/bash
# Utility functions for managing background job monitors
# Usage: bash monitor_utils.sh [command] [args...]

show_help() {
    echo "Background Monitor Utilities"
    echo ""
    echo "Commands:"
    echo "  list                    - Show all active background monitors"
    echo "  status JOB_ID           - Show status of specific job monitor"
    echo "  logs JOB_ID             - Follow logs of specific job monitor"
    echo "  stop JOB_ID             - Stop background monitor for specific job"
    echo "  sync                    - Manual WandB sync of all offline runs"
    echo "  cleanup                 - Remove old monitor logs (older than 7 days)"
    echo ""
    echo "Examples:"
    echo "  bash monitor_utils.sh list"
    echo "  bash monitor_utils.sh status 12345"
    echo "  bash monitor_utils.sh logs 12345"
    echo "  bash monitor_utils.sh stop 12345"
}

list_monitors() {
    echo "🔍 Active Background Monitors:"
    echo "=============================="
    
    # Find all background monitor processes
    MONITORS=$(pgrep -f "background_monitor.sh" 2>/dev/null || true)
    
    if [[ -z "$MONITORS" ]]; then
        echo "No active background monitors found."
        return
    fi
    
    echo "PID    | Job ID | Started"
    echo "-------|--------|--------"
    
    for PID in $MONITORS; do
        # Get process details
        CMDLINE=$(ps -p "$PID" -o args --no-headers 2>/dev/null || echo "N/A")
        if [[ "$CMDLINE" =~ background_monitor\.sh[[:space:]]+([0-9]+) ]]; then
            JOB_ID="${BASH_REMATCH[1]}"
            START_TIME=$(ps -p "$PID" -o lstart --no-headers 2>/dev/null | cut -c 1-15 || echo "Unknown")
            echo "$PID | $JOB_ID | $START_TIME"
        fi
    done
    
    echo ""
    echo "📝 Recent monitor logs:"
    find logs -name "background_monitor_*.log" -mtime -1 2>/dev/null | sort -r | head -5 | while read -r log; do
        TIMESTAMP=$(stat -c %y "$log" 2>/dev/null | cut -d'.' -f1)
        echo "   $log (modified: $TIMESTAMP)"
    done
}

show_status() {
    local JOB_ID="$1"
    if [[ -z "$JOB_ID" ]]; then
        echo "Error: Job ID required"
        echo "Usage: bash monitor_utils.sh status JOB_ID"
        exit 1
    fi
    
    echo "📊 Status for Job $JOB_ID:"
    echo "========================"
    
    # Check if job is in queue
    if squeue -h -j "$JOB_ID" >/dev/null 2>&1; then
        echo "🟢 SLURM Job: RUNNING"
        squeue -j "$JOB_ID"
    else
        echo "🔴 SLURM Job: NOT RUNNING"
        # Try to get completion status
        STATUS=$(sacct -j "$JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
        if [[ -n "$STATUS" ]]; then
            echo "   Final status: $STATUS"
        fi
    fi
    
    # Check if monitor is running
    MONITOR_PID=$(pgrep -f "background_monitor.sh $JOB_ID" 2>/dev/null || true)
    if [[ -n "$MONITOR_PID" ]]; then
        echo "🟢 Background Monitor: RUNNING (PID: $MONITOR_PID)"
    else
        echo "🔴 Background Monitor: NOT RUNNING"
    fi
    
    # Show log tail
    LOG_FILE="logs/background_monitor_${JOB_ID}.log"
    if [[ -f "$LOG_FILE" ]]; then
        echo ""
        echo "📝 Recent monitor activity (last 5 lines):"
        tail -5 "$LOG_FILE"
    else
        echo ""
        echo "❌ Monitor log not found: $LOG_FILE"
    fi
}

follow_logs() {
    local JOB_ID="$1"
    if [[ -z "$JOB_ID" ]]; then
        echo "Error: Job ID required"
        echo "Usage: bash monitor_utils.sh logs JOB_ID"
        exit 1
    fi
    
    LOG_FILE="logs/background_monitor_${JOB_ID}.log"
    if [[ -f "$LOG_FILE" ]]; then
        echo "📝 Following logs for Job $JOB_ID (Ctrl+C to exit):"
        echo "=================================================="
        tail -f "$LOG_FILE"
    else
        echo "❌ Monitor log not found: $LOG_FILE"
        exit 1
    fi
}

stop_monitor() {
    local JOB_ID="$1"
    if [[ -z "$JOB_ID" ]]; then
        echo "Error: Job ID required"
        echo "Usage: bash monitor_utils.sh stop JOB_ID"
        exit 1
    fi
    
    MONITOR_PID=$(pgrep -f "background_monitor.sh $JOB_ID" 2>/dev/null || true)
    if [[ -n "$MONITOR_PID" ]]; then
        echo "🛑 Stopping background monitor for Job $JOB_ID (PID: $MONITOR_PID)..."
        kill "$MONITOR_PID"
        sleep 2
        
        # Verify it stopped
        if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
            echo "✅ Monitor stopped successfully"
        else
            echo "⚠️  Force killing monitor..."
            kill -9 "$MONITOR_PID" 2>/dev/null || true
        fi
    else
        echo "❌ No background monitor found for Job $JOB_ID"
    fi
}

manual_sync() {
    echo "🔄 Manual WandB sync of all offline runs..."
    
    if [[ -f sc_venv_template/activate.sh ]]; then
        (
            set +e
            source sc_venv_template/activate.sh
            if python -c 'import wandb; print("wandb-ok")' >/dev/null 2>&1; then
                echo "☁️  Syncing wandb offline runs..."
                
                OFFLINE_RUNS=$(find wandb -name "offline-run-*" -type d 2>/dev/null | wc -l)
                OFFLINE_RUNS_NESTED=$(find wandb -path "*/offline*/*" -type d 2>/dev/null | wc -l)
                
                if [[ $OFFLINE_RUNS -gt 0 ]] || [[ $OFFLINE_RUNS_NESTED -gt 0 ]]; then
                    echo "📊 Found $OFFLINE_RUNS top-level and $OFFLINE_RUNS_NESTED nested offline runs"
                    
                    WANDB_MODE=run WANDB_CONSOLE=off python -m wandb sync wandb/offline-run-* 2>/dev/null || true
                    WANDB_MODE=run WANDB_CONSOLE=off python -m wandb sync wandb/offline*/* 2>/dev/null || true
                    
                    echo "✅ Manual WandB sync completed"
                else
                    echo "📭 No offline runs found to sync"
                fi
            else
                echo "⚠️  wandb not available in environment"
            fi
        )
    else
        echo "⚠️  sc_venv_template/activate.sh not found"
    fi
}

cleanup_logs() {
    echo "🧹 Cleaning up old monitor logs (older than 7 days)..."
    
    OLD_LOGS=$(find logs -name "background_monitor_*.log" -mtime +7 2>/dev/null || true)
    if [[ -n "$OLD_LOGS" ]]; then
        echo "$OLD_LOGS" | while read -r log; do
            echo "   Removing: $log"
            rm -f "$log"
        done
        echo "✅ Cleanup completed"
    else
        echo "📭 No old logs found"
    fi
}

# Main command dispatcher
case "$1" in
    list|ls)
        list_monitors
        ;;
    status)
        show_status "$2"
        ;;
    logs|log)
        follow_logs "$2"
        ;;
    stop|kill)
        stop_monitor "$2"
        ;;
    sync)
        manual_sync
        ;;
    cleanup|clean)
        cleanup_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
