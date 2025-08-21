#!/bin/bash
#SBATCH --job-name=arena_rewards
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=320GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --output=logs/arena_rewards_%j.out
#SBATCH --error=logs/arena_rewards_%j.err

# Arena Reward Comparison Experiment - RSL-RL Integrated
# This script runs two training jobs comparing sparse vs dense rewards
# Designed to finish in ~10 minutes on 96-core A100 node with comprehensive logging

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the cluster virtual environment  
cd sc_venv_template
source activate.sh
cd ..

echo "🚀 Starting Arena Reward Comparison Experiment"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "======================================================="

# Ensure output directories exist
mkdir -p logs
mkdir -p models

# Force wandb into silent offline mode (prevents retry loops)
export WANDB_MODE="offline"
export WANDB_CONSOLE="off"
export WANDB_SILENT="true"

# Common training parameters (optimized for 10-minute run on A100 + 96 cores)
# Each experiment gets 32 envs, ~10 minutes runtime with RSL-RL integration
COMMON_ARGS="--env_name pointmaze-arena-v0 \
--num_envs 32 \
--total_timesteps 200000 \
--include_goal \
--config config/ogbench_config.yaml"

# Launch both experiments in parallel
echo ""
echo "🚀 Launching SPARSE and DENSE reward experiments in parallel..."
echo "========================================================"

# Experiment 1: SPARSE Rewards (Goal-only) - Background execution
echo "📊 Starting SPARSE reward training..."
python train_rsl_rl_integrated.py $COMMON_ARGS \
    --reward_type sparse \
    --experiment_name "arena_sparse_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/arena_sparse_$SLURM_JOB_ID.log &
PID_SPARSE=$!

# Experiment 2: DENSE Rewards (Distance-based) - Background execution  
echo "📊 Starting DENSE reward training..."
python train_rsl_rl_integrated.py $COMMON_ARGS \
    --reward_type dense \
    --dense_reward_scale 0.01 \
    --experiment_name "arena_dense_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/arena_dense_$SLURM_JOB_ID.log &
PID_DENSE=$!

echo "✓ Both training jobs launched in parallel"
echo "Process IDs: Sparse=$PID_SPARSE, Dense=$PID_DENSE"
echo ""

# Wait for all jobs to complete with error checking
echo "⏳ Waiting for parallel training jobs to complete..."

wait $PID_SPARSE
SPARSE_EXIT_CODE=$?
if [ $SPARSE_EXIT_CODE -eq 0 ]; then
    echo "✅ Sparse reward training completed successfully"
else
    echo "❌ Sparse reward training failed (exit code: $SPARSE_EXIT_CODE)"
fi

wait $PID_DENSE
DENSE_EXIT_CODE=$?
if [ $DENSE_EXIT_CODE -eq 0 ]; then
    echo "✅ Dense reward training completed successfully"
else
    echo "❌ Dense reward training failed (exit code: $DENSE_EXIT_CODE)"
fi

# Overall success check
if [ $SPARSE_EXIT_CODE -ne 0 ] || [ $DENSE_EXIT_CODE -ne 0 ]; then
    echo "❌ One or more experiments failed!"
    exit 1
fi

echo ""
echo "✅ Both experiments completed successfully!"
echo "Time: $(date)"

# Summary of results
echo ""
echo "📈 Experiment Summary"
echo "========================================================"

# Find the latest RSL-RL model directories
SPARSE_DIR=$(find logs/rsl_rl -name "*arena_sparse*" -type d | sort | tail -1)
DENSE_DIR=$(find logs/rsl_rl -name "*arena_dense*" -type d | sort | tail -1)

echo "Sparse reward results:"
if [ -d "$SPARSE_DIR" ]; then
    echo "  RSL-RL log directory: $SPARSE_DIR"
    echo "  Models saved:"
    ls -la "$SPARSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Recent training output:"
    tail -5 "logs/arena_sparse_$SLURM_JOB_ID.log" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No sparse model directory found"
fi

echo ""
echo "Dense reward results:"
if [ -d "$DENSE_DIR" ]; then
    echo "  RSL-RL log directory: $DENSE_DIR"
    echo "  Models saved:"
    ls -la "$DENSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Recent training output:"
    tail -5 "logs/arena_dense_$SLURM_JOB_ID.log" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No dense model directory found"
fi

echo ""
echo "💾 Logs and models saved in:"
echo "  - Training logs: logs/"
echo "  - RSL-RL models: logs/rsl_rl/"
echo "  - Wandb: wandb/ (use 'wandb sync' to upload)"

echo ""
echo "🎯 Next steps:"
echo "  1. Check goal success rates in the training logs above"
echo "  2. Sync wandb logs: wandb sync wandb/"
echo "  3. Compare /Episode/goal_success_rate metrics between sparse vs dense"
echo "  4. Test model: python eval_interactive.py (if evaluation script exists)"

echo "========================================================"
echo "🏁 Arena Reward Comparison Complete - $(date)"