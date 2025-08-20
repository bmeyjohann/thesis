#!/bin/bash
#SBATCH --job-name=arena_rewards
#SBATCH --account=<your_account>
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=logs/arena_rewards_%j.out
#SBATCH --error=logs/arena_rewards_%j.err

# Arena Reward Comparison Experiment
# This script runs two training jobs comparing sparse vs dense rewards
# Designed to finish in under 1 hour with comprehensive logging

# Load modules
module purge
module load Stages/2024
module load GCC/13.2.0
module load Python/3.11.5
module load CUDA/12

echo "🚀 Starting Arena Reward Comparison Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "========================================================"

# Environment setup
cd $SLURM_SUBMIT_DIR
source sc_venv_template/venv/bin/activate

# Ensure output directories exist
mkdir -p logs
mkdir -p models

# Set environment variables for offline wandb
export WANDB_MODE=offline
export WANDB_DIR=$PWD/wandb

# Common training parameters (optimized for A100 + 96 cores)
COMMON_ARGS="--env_name pointmaze-arena-v0 \
--include_goal \
--num_envs 32 \
--total_timesteps 1000000 \
--num_steps_per_env 1024 \
--learning_rate 3e-4 \
--num_learning_epochs 8 \
--num_mini_batches 16 \
--save_interval 25 \
--use_wandb \
--wandb_project arena-reward-comparison"

echo ""
echo "📊 Experiment 1: SPARSE Rewards (Goal-only)"
echo "========================================================"
python train_unified.py $COMMON_ARGS \
    --reward_type sparse \
    --experiment_name "arena_sparse_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/arena_sparse_$SLURM_JOB_ID.log

# Check if first experiment succeeded
if [ $? -ne 0 ]; then
    echo "❌ Sparse reward experiment failed!"
    exit 1
fi

echo ""
echo "📊 Experiment 2: DENSE Rewards (Distance-based)"
echo "========================================================"
python train_unified.py $COMMON_ARGS \
    --reward_type dense \
    --dense_reward_scale 0.01 \
    --experiment_name "arena_dense_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/arena_dense_$SLURM_JOB_ID.log

# Check if second experiment succeeded
if [ $? -ne 0 ]; then
    echo "❌ Dense reward experiment failed!"
    exit 1
fi

echo ""
echo "✅ Both experiments completed successfully!"
echo "Time: $(date)"

# Summary of results
echo ""
echo "📈 Experiment Summary"
echo "========================================================"

# Find the latest model directories
SPARSE_DIR=$(find models -name "*arena_sparse*" -type d | sort | tail -1)
DENSE_DIR=$(find models -name "*arena_dense*" -type d | sort | tail -1)

echo "Sparse reward results:"
if [ -d "$SPARSE_DIR" ]; then
    echo "  Model directory: $SPARSE_DIR"
    echo "  Models saved:"
    ls -la "$SPARSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Training log:"
    tail -3 "$SPARSE_DIR/training_log.csv" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No sparse model directory found"
fi

echo ""
echo "Dense reward results:"
if [ -d "$DENSE_DIR" ]; then
    echo "  Model directory: $DENSE_DIR"
    echo "  Models saved:"
    ls -la "$DENSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Training log:"
    tail -3 "$DENSE_DIR/training_log.csv" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No dense model directory found"
fi

echo ""
echo "💾 Logs and models saved in:"
echo "  - Logs: logs/"
echo "  - Models: models/"
echo "  - Wandb: wandb/ (use 'wandb sync' to upload)"

echo ""
echo "🎯 Next steps:"
echo "  1. Check model performance: python eval_interactive.py --model_path models/.../best_model.pt --env_name pointmaze-arena-v0"
echo "  2. Sync wandb logs: wandb sync wandb/"
echo "  3. Compare sparse vs dense success rates in the training logs"

echo "========================================================"
echo "🏁 Arena Reward Comparison Complete - $(date)"