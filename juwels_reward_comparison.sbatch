#!/bin/bash
#SBATCH --account=<your_account>  # Replace with your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --job-name=ogbench_reward_comparison
#SBATCH --output=logs/reward_comparison_%j.out
#SBATCH --error=logs/reward_comparison_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the cluster virtual environment  
cd sc_venv_template
source activate.sh
cd ..

echo "🚀 Starting OGBench Reward Comparison Study on JUWELS"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

# Define shared training parameters (LIGHTER CONFIGURATION)
SHARED_ARGS="--num_envs 16 --log_interval 25 --use_wandb --wandb_project juwels-reward-pilot"

# Run 3 parallel training jobs on 3 GPUs (BUDGET-CONSCIOUS)
echo "📊 Launching 3 lightweight reward shaping pilot experiments..."

# GPU 0: Sparse Rewards (Baseline) - Reduced to 5M steps
echo "GPU 0: Starting SPARSE reward training (5M steps)..."
CUDA_VISIBLE_DEVICES=0 python train_rsl_rl_clean.py \
    --reward_type sparse \
    --total_timesteps 5000000 \
    --max_iterations 312 \
    --experiment_name "juwels_sparse_5M_pilot_$(date +%Y%m%d_%H%M)" \
    $SHARED_ARGS &
PID1=$!

# GPU 1: Dense Rewards - Higher scaling + 3M steps  
echo "GPU 1: Starting DENSE reward training (3M steps, higher scaling)..."
CUDA_VISIBLE_DEVICES=1 python train_rsl_rl_clean.py \
    --reward_type dense \
    --dense_reward_scale 0.1 \
    --total_timesteps 3000000 \
    --max_iterations 187 \
    --experiment_name "juwels_dense_3M_pilot_$(date +%Y%m%d_%H%M)" \
    $SHARED_ARGS &
PID2=$!

# GPU 2: Mixed Rewards - 4M steps
echo "GPU 2: Starting MIXED reward training (4M steps)..."  
CUDA_VISIBLE_DEVICES=2 python train_rsl_rl_clean.py \
    --reward_type mixed \
    --dense_reward_scale 0.05 \
    --mixed_dense_weight 0.2 \
    --total_timesteps 4000000 \
    --max_iterations 250 \
    --experiment_name "juwels_mixed_4M_pilot_$(date +%Y%m%d_%H%M)" \
    $SHARED_ARGS &
PID3=$!

echo "✓ All training jobs launched"
echo "Process IDs: Sparse=$PID1, Dense=$PID2, Mixed=$PID3"
echo "Monitor progress at: https://wandb.ai/<your_username>/juwels-reward-study"

# Wait for all jobs to complete
wait $PID1
echo "✓ Sparse training completed"

wait $PID2  
echo "✓ Dense training completed"

wait $PID3
echo "✓ Mixed training completed"

echo "🎉 All reward shaping experiments completed!"
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

# Optional: Generate comparison summary
echo "📈 Pilot Training Summary:"
echo "Sparse (5M steps):  Expected ~1 hour"
echo "Dense (3M steps):   Expected ~40 minutes" 
echo "Mixed (4M steps):   Expected ~50 minutes"
echo "Total budget used:  ~50 core-hours (0.08% of total)"
echo ""
echo "📊 Results available in:"
echo "  - Wandb dashboard: https://wandb.ai/<username>/juwels-reward-pilot"
echo "  - CSV logs: ./logs/*_metrics.csv"
echo "  - Model files: ./models/*/"
