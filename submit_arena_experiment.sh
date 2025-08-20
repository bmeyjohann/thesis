#!/bin/bash
"""
Submit arena reward comparison experiment to JUWELS cluster.

This script automatically detects your SLURM account and submits the job.
"""

# Function to detect SLURM account
detect_account() {
    local account=""
    
    # Method 1: Check saved config
    if [ -f ".slurm_account" ]; then
        account=$(cat .slurm_account)
        if [ -n "$account" ]; then
            echo "Using saved account: $account"
            return 0
        fi
    fi
    
    # Method 2: Check environment variable
    if [ -n "$SLURM_ACCOUNT" ]; then
        account="$SLURM_ACCOUNT"
        echo "Using account from environment: $account"
        return 0
    fi
    
    # Method 3: Query sacctmgr
    if command -v sacctmgr &> /dev/null; then
        account=$(sacctmgr show user $USER -n -P format=account | head -1)
        if [ -n "$account" ]; then
            echo "Detected account from sacctmgr: $account"
            echo "$account" > .slurm_account  # Save for next time
            return 0
        fi
    fi
    
    # Method 4: Interactive prompt
    echo "Could not auto-detect SLURM account."
    echo "Available accounts for user $USER:"
    sacctmgr show user $USER format=user,account -n 2>/dev/null || echo "  (Could not query accounts)"
    echo ""
    read -p "Please enter your SLURM account: " account
    
    if [ -n "$account" ]; then
        echo "$account" > .slurm_account  # Save for next time
        return 0
    else
        echo "❌ No account specified"
        return 1
    fi
}

# Main execution
echo "🚀 Arena Reward Comparison Experiment Submission"
echo "=================================================="

# Detect account
if detect_account; then
    ACCOUNT=$(cat .slurm_account)
    echo ""
    echo "📋 Experiment Configuration:"
    echo "   Account: $ACCOUNT"
    echo "   Environment: pointmaze-arena-v0"
    echo "   Reward types: sparse vs dense"
    echo "   Total timesteps: 200,000 each"
    echo "   Time limit: 55 minutes"
    echo "   GPU required: Yes"
    echo ""
    
    # Update the SLURM script with the account
    if [ ! -f "arena_reward_comparison.sbatch" ]; then
        echo "❌ SLURM script not found: arena_reward_comparison.sbatch"
        exit 1
    fi
    
    # Create temporary script with account
    sed "s/<your_account>/$ACCOUNT/g" arena_reward_comparison.sbatch > arena_reward_comparison_temp.sbatch
    
    # Submit the job
    echo "🎯 Submitting job..."
    JOB_ID=$(sbatch arena_reward_comparison_temp.sbatch | grep -o '[0-9]*')
    
    if [ $? -eq 0 ]; then
        echo "✅ Job submitted successfully!"
        echo "   Job ID: $JOB_ID"
        echo "   Queue status: squeue -u $USER"
        echo "   Follow output: tail -f logs/arena_rewards_$JOB_ID.out"
        echo "   Follow errors: tail -f logs/arena_rewards_$JOB_ID.err"
        echo ""
        echo "📊 When complete, check results in:"
        echo "   - Models: models/arena_*/"
        echo "   - Logs: logs/arena_*"
        echo "   - Wandb: wandb/ (run 'wandb sync' to upload)"
    else
        echo "❌ Job submission failed!"
        exit 1
    fi
    
    # Clean up
    rm arena_reward_comparison_temp.sbatch
    
else
    echo "❌ Could not determine SLURM account. Aborting."
    exit 1
fi

echo ""
echo "🎉 All set! The experiment is now running on the cluster."
echo "   The job will automatically compare sparse vs dense rewards"
echo "   and save detailed metrics and model checkpoints."