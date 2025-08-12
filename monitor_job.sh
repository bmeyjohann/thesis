#!/bin/bash
# Monitor SLURM job constraints and status

USER_JOBS=$(squeue -u $USER -h -o %i)

if [[ -z "$USER_JOBS" ]]; then
    echo "❌ No active jobs found for user: $USER"
    echo ""
    echo "📊 Recent completed jobs (last 24h):"
    sacct -u $USER -S $(date -d '1 day ago' +%Y-%m-%d) --format=JobID,JobName,State,ElapsedTime,AllocCPUS,AllocGRES,Timelimit
    exit 1
fi

echo "🔍 Active Job Monitoring for: $USER"
echo "=================================="

for JOB_ID in $USER_JOBS; do
    echo ""
    echo "📋 Job ID: $JOB_ID"
    echo "-------------------"
    
    # Get basic info
    BASIC_INFO=$(squeue -j $JOB_ID -h -o "%j|%T|%M|%L|%D|%C|%G|%P")
    IFS='|' read -r JOB_NAME STATE RUNTIME TIMELEFT NODES CPUS GRES PARTITION <<< "$BASIC_INFO"
    
    echo "Name:       $JOB_NAME"
    echo "State:      $STATE"
    echo "Runtime:    $RUNTIME"
    echo "Time Left:  $TIMELEFT"
    echo "Partition:  $PARTITION"
    echo "Nodes:      $NODES"
    echo "CPUs:       $CPUS"
    echo "GPUs:       $GRES"
    
    # Get detailed constraints from scontrol
    echo ""
    echo "🔧 Detailed Constraints:"
    scontrol show job $JOB_ID | grep -E "(TimeLimit|CPUs|Gres|Partition|RunTime|StartTime|EndTime)" | sed 's/^/   /'
    
    # Check if job matches expected resources
    echo ""
    echo "✅ Constraint Verification:"
    
    # Expected values for our jobs
    if [[ "$JOB_NAME" == "ogbench_reward_comparison" ]]; then
        EXPECTED_CPUS=32
        EXPECTED_GPUS=3
        EXPECTED_TIME="02:00:00"
    elif [[ "$JOB_NAME" == "ogbench_test" ]]; then
        EXPECTED_CPUS=8
        EXPECTED_GPUS=1
        EXPECTED_TIME="00:10:00"
    else
        echo "   ⚠️  Unknown job name - cannot verify constraints"
        continue
    fi
    
    # Verify CPUs
    if [[ "$CPUS" == "$EXPECTED_CPUS" ]]; then
        echo "   ✅ CPUs: $CPUS (matches expected $EXPECTED_CPUS)"
    else
        echo "   ❌ CPUs: $CPUS (expected $EXPECTED_CPUS)"
    fi
    
    # Verify GPUs (extract number from gres string like "gpu:3")
    GPU_COUNT=$(echo "$GRES" | grep -o '[0-9]\+' | head -1)
    if [[ "$GPU_COUNT" == "$EXPECTED_GPUS" ]]; then
        echo "   ✅ GPUs: $GPU_COUNT (matches expected $EXPECTED_GPUS)"
    else
        echo "   ❌ GPUs: $GPU_COUNT (expected $EXPECTED_GPUS)"
    fi
    
    # Get actual time limit from scontrol for precise check
    ACTUAL_TIME=$(scontrol show job $JOB_ID | grep -o 'TimeLimit=[^ ]*' | cut -d'=' -f2)
    if [[ "$ACTUAL_TIME" == "$EXPECTED_TIME" ]]; then
        echo "   ✅ Time Limit: $ACTUAL_TIME (matches expected $EXPECTED_TIME)"
    else
        echo "   ❌ Time Limit: $ACTUAL_TIME (expected $EXPECTED_TIME)"
    fi
    
    echo ""
    echo "📊 Resource Usage (if running):"
    if [[ "$STATE" == "RUNNING" ]]; then
        # Try to get resource usage
        sstat -j $JOB_ID --format=AveCPU,AveRSS,MaxRSS 2>/dev/null | tail -n +3 | sed 's/^/   /'
    else
        echo "   (Job not yet running)"
    fi
    
done

echo ""
echo "🔄 Monitor commands:"
echo "  watch -n 5 'squeue -u $USER'"
echo "  tail -f logs/reward_comparison_*.out"
echo "  bash monitor_job.sh"

