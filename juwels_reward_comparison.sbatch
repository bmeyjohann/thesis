#!/bin/bash
#SBATCH --account=<your_account>  # This will be auto-replaced by submit_job.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --job-name=ogbench_mixed_rewards
#SBATCH --output=logs/reward_comparison_%j.out
#SBATCH --error=logs/reward_comparison_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the cluster virtual environment  
cd sc_venv_template
source activate.sh
cd ..

# Force wandb into silent offline mode (prevents retry loops)
export WANDB_MODE="offline"
export WANDB_CONSOLE="off" 
export WANDB_SILENT="true"
export WANDB__SERVICE_WAIT="0"

echo "🚀 Starting OGBench Reward Comparison Study on JUWELS"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

# Define shared training parameters (LIGHTER CONFIGURATION)
SHARED_ARGS="--num_envs 16 --log_interval 25 --wandb_project juwels-mixed-rewards --use_wandb"

# Run 3 parallel training jobs on 1 A100 GPU (plenty of memory for small RL models)
echo "📊 Launching 3 parallel mixed reward comparison experiments..."

# Job 1: Dense Rewards (Baseline comparison) 
echo "🎯 Starting DENSE reward training (baseline)..."
python train_rsl_rl_clean.py \
    --reward_type dense \
    --dense_reward_scale 0.1 \
    --total_timesteps 15000000 \
    --max_iterations 750 \
    --experiment_name "juwels_dense_baseline_$(date +%Y%m%d_%H%M)" \
    $SHARED_ARGS &
PID1=$!

# Job 2: Mixed Rewards - Original weighting (higher dense influence)
echo "⚖️  Starting MIXED reward training (original weighting)..."  
python train_rsl_rl_clean.py \
    --reward_type mixed \
    --dense_reward_scale 0.05 \
    --mixed_dense_weight 0.2 \
    --total_timesteps 15000000 \
    --max_iterations 750 \
    --experiment_name "juwels_mixed_high_$(date +%Y%m%d_%H%M)" \
    $SHARED_ARGS &
PID2=$!

# Job 3: Mixed Rewards - Lower dense influence
echo "🔽 Starting MIXED reward training (lower dense weight)..."  
python train_rsl_rl_clean.py \
    --reward_type mixed \
    --dense_reward_scale 0.05 \
    --mixed_dense_weight 0.05 \
    --total_timesteps 15000000 \
    --max_iterations 750 \
    --experiment_name "juwels_mixed_low_$(date +%Y%m%d_%H%M)" \
    $SHARED_ARGS &
PID3=$!

echo "✓ All training jobs launched in parallel"
echo "Process IDs: Dense=$PID1, Mixed_High=$PID2, Mixed_Low=$PID3"
echo "Monitor progress at: https://wandb.ai/<your_username>/juwels-mixed-rewards"

# Wait for all jobs to complete with error checking
echo "⏳ Waiting for parallel training jobs to complete..."

wait $PID1
if [ $? -eq 0 ]; then
    echo "✅ Dense baseline training completed successfully"
else
    echo "❌ Dense baseline training failed (exit code: $?)"
fi

wait $PID2  
if [ $? -eq 0 ]; then
    echo "✅ Mixed high weight training completed successfully"
else
    echo "❌ Mixed high weight training failed (exit code: $?)"
fi

wait $PID3
if [ $? -eq 0 ]; then
    echo "✅ Mixed low weight training completed successfully"
else
    echo "❌ Mixed low weight training failed (exit code: $?)"
fi

echo "🎉 All mixed reward comparison experiments completed!"
date

# Sync wandb runs from compute node (if they were run in offline mode)
echo "📤 Syncing wandb runs to cloud..."
if [ -d "$HOME/.wandb" ]; then
    # Find and sync all offline runs
    find $HOME/.wandb -name "offline-run-*" -type d | while read run_dir; do
        echo "Syncing: $run_dir"
        wandb sync "$run_dir" || echo "⚠️  Failed to sync $run_dir"
    done
    
    # Also check for runs in the working directory
    find wandb -name "offline-run-*" -type d 2>/dev/null | while read run_dir; do
        echo "Syncing: $run_dir"
        wandb sync "$run_dir" || echo "⚠️  Failed to sync $run_dir"
    done
else
    echo "No offline wandb runs found to sync"
fi

# Generate comparison summary
echo "📈 Mixed Reward Comparison Summary:"
echo "Dense baseline (15M steps):     Expected ~1.2 hours"
echo "Mixed high weight (15M steps):  Expected ~1.2 hours" 
echo "Mixed low weight (15M steps):   Expected ~1.2 hours"
echo "Total runtime: ~1.2 hours (parallel on 1 A100 GPU)"
echo "Total budget used: ~19 core-hours (0.03% of total)"
echo ""
echo "🔬 Experiment Design:"
echo "  Dense baseline:    dense_reward_scale=0.1"
echo "  Mixed high:        dense_scale=0.05, weight=0.2 (20% dense influence)"  
echo "  Mixed low:         dense_scale=0.05, weight=0.05 (5% dense influence)"
echo ""
echo "📊 Results available in:"
echo "  - Wandb dashboard: https://wandb.ai/<username>/juwels-mixed-rewards"
echo "  - CSV logs: ./logs/*_metrics.csv"
echo "  - Model files: ./models/*/"
