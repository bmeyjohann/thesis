#!/bin/bash
#SBATCH --job-name=maze_env_reward_comp
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=480GB
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=logs/maze_env_reward_comp_%j.out
#SBATCH --error=logs/maze_env_reward_comp_%j.err

# Maze Environment & Reward Comparison Experiment - RSL-RL Integrated
# This script runs four training jobs comparing environments and reward types:
# 1. Arena + Sparse, 2. Arena + Dense, 3. Medium + Sparse, 4. Medium + Dense
# Designed to finish in ~2 hours on high-performance node with comprehensive logging

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the cluster virtual environment  
cd sc_venv_template
source activate.sh
cd ..

echo "🚀 Starting Maze Environment & Reward Comparison Experiment"
echo "Experiments: Arena+Sparse, Arena+Dense, Medium+Sparse, Medium+Dense"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "======================================================="

# Ensure output directories exist
mkdir -p logs
mkdir -p models

# Force wandb into silent offline mode (prevents retry loops)
# Useless as this code runs on the login node
# export WANDB_MODE="offline"
# export WANDB_CONSOLE="off"
# export WANDB_SILENT="true"

# Common training parameters (optimized for 4-experiment parallel run)
# Each experiment gets 32 envs to balance resource usage across 4 jobs
# Total: 4 experiments × 32 envs = 128 parallel environments
BASE_ARGS="--num_envs 64 \
--total_timesteps 5000000 \
--include_goal \
--config config/ogbench_config.yaml"

# Launch all four experiments in parallel
echo ""
echo "🚀 Launching all 4 experiments in parallel..."
echo "========================================================"

# Experiment 1: Arena + Sparse Rewards
echo "📊 Starting ARENA + SPARSE reward training..."
python train_rsl_rl_integrated.py $BASE_ARGS \
    --env_name pointmaze-arena-v0 \
    --reward_type sparse \
    --experiment_name "arena_sparse_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/arena_sparse_$SLURM_JOB_ID.log &
PID_ARENA_SPARSE=$!

# Experiment 2: Arena + Dense Rewards
echo "📊 Starting ARENA + DENSE reward training..."
python train_rsl_rl_integrated.py $BASE_ARGS \
    --env_name pointmaze-arena-v0 \
    --reward_type dense \
    --dense_reward_scale 0.01 \
    --experiment_name "arena_dense_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/arena_dense_$SLURM_JOB_ID.log &
PID_ARENA_DENSE=$!

# Experiment 3: Medium + Sparse Rewards
echo "📊 Starting MEDIUM + SPARSE reward training..."
python train_rsl_rl_integrated.py $BASE_ARGS \
    --env_name pointmaze-medium-v0 \
    --reward_type sparse \
    --experiment_name "medium_sparse_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/medium_sparse_$SLURM_JOB_ID.log &
PID_MEDIUM_SPARSE=$!

# Experiment 4: Medium + Dense Rewards  
echo "📊 Starting MEDIUM + DENSE reward training..."
python train_rsl_rl_integrated.py $BASE_ARGS \
    --env_name pointmaze-medium-v0 \
    --reward_type dense \
    --dense_reward_scale 0.01 \
    --experiment_name "medium_dense_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/medium_dense_$SLURM_JOB_ID.log &
PID_MEDIUM_DENSE=$!

echo "✓ All 4 training jobs launched in parallel"
echo "Process IDs:"
echo "  Arena+Sparse: $PID_ARENA_SPARSE"
echo "  Arena+Dense:  $PID_ARENA_DENSE" 
echo "  Medium+Sparse: $PID_MEDIUM_SPARSE"
echo "  Medium+Dense:  $PID_MEDIUM_DENSE"
echo ""

# Wait for all jobs to complete with error checking
echo "⏳ Waiting for all 4 parallel training jobs to complete..."

wait $PID_ARENA_SPARSE
ARENA_SPARSE_EXIT_CODE=$?
if [ $ARENA_SPARSE_EXIT_CODE -eq 0 ]; then
    echo "✅ Arena + Sparse reward training completed successfully"
else
    echo "❌ Arena + Sparse reward training failed (exit code: $ARENA_SPARSE_EXIT_CODE)"
fi

wait $PID_ARENA_DENSE
ARENA_DENSE_EXIT_CODE=$?
if [ $ARENA_DENSE_EXIT_CODE -eq 0 ]; then
    echo "✅ Arena + Dense reward training completed successfully"
else
    echo "❌ Arena + Dense reward training failed (exit code: $ARENA_DENSE_EXIT_CODE)"
fi

wait $PID_MEDIUM_SPARSE
MEDIUM_SPARSE_EXIT_CODE=$?
if [ $MEDIUM_SPARSE_EXIT_CODE -eq 0 ]; then
    echo "✅ Medium + Sparse reward training completed successfully"
else
    echo "❌ Medium + Sparse reward training failed (exit code: $MEDIUM_SPARSE_EXIT_CODE)"
fi

wait $PID_MEDIUM_DENSE
MEDIUM_DENSE_EXIT_CODE=$?
if [ $MEDIUM_DENSE_EXIT_CODE -eq 0 ]; then
    echo "✅ Medium + Dense reward training completed successfully"
else
    echo "❌ Medium + Dense reward training failed (exit code: $MEDIUM_DENSE_EXIT_CODE)"
fi

# Overall success check
if [ $ARENA_SPARSE_EXIT_CODE -ne 0 ] || [ $ARENA_DENSE_EXIT_CODE -ne 0 ] || [ $MEDIUM_SPARSE_EXIT_CODE -ne 0 ] || [ $MEDIUM_DENSE_EXIT_CODE -ne 0 ]; then
    echo "❌ One or more experiments failed!"
    exit 1
fi

echo ""
echo "✅ All 4 experiments completed successfully!"
echo "Time: $(date)"

# Summary of results
echo ""
echo "📈 Experiment Summary"
echo "========================================================"

# Find the latest RSL-RL model directories for all experiments
ARENA_SPARSE_DIR=$(find logs/rsl_rl -name "*arena_sparse*" -type d | sort | tail -1)
ARENA_DENSE_DIR=$(find logs/rsl_rl -name "*arena_dense*" -type d | sort | tail -1)
MEDIUM_SPARSE_DIR=$(find logs/rsl_rl -name "*medium_sparse*" -type d | sort | tail -1)
MEDIUM_DENSE_DIR=$(find logs/rsl_rl -name "*medium_dense*" -type d | sort | tail -1)

echo "1. Arena + Sparse reward results:"
if [ -d "$ARENA_SPARSE_DIR" ]; then
    echo "  RSL-RL log directory: $ARENA_SPARSE_DIR"
    echo "  Models saved:"
    ls -la "$ARENA_SPARSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Recent training output:"
    tail -5 "logs/arena_sparse_$SLURM_JOB_ID.log" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No arena sparse model directory found"
fi

echo ""
echo "2. Arena + Dense reward results:"
if [ -d "$ARENA_DENSE_DIR" ]; then
    echo "  RSL-RL log directory: $ARENA_DENSE_DIR"
    echo "  Models saved:"
    ls -la "$ARENA_DENSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Recent training output:"
    tail -5 "logs/arena_dense_$SLURM_JOB_ID.log" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No arena dense model directory found"
fi

echo ""
echo "3. Medium + Sparse reward results:"
if [ -d "$MEDIUM_SPARSE_DIR" ]; then
    echo "  RSL-RL log directory: $MEDIUM_SPARSE_DIR"
    echo "  Models saved:"
    ls -la "$MEDIUM_SPARSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Recent training output:"
    tail -5 "logs/medium_sparse_$SLURM_JOB_ID.log" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No medium sparse model directory found"
fi

echo ""
echo "4. Medium + Dense reward results:"
if [ -d "$MEDIUM_DENSE_DIR" ]; then
    echo "  RSL-RL log directory: $MEDIUM_DENSE_DIR"
    echo "  Models saved:"
    ls -la "$MEDIUM_DENSE_DIR"/*.pt 2>/dev/null || echo "    No model files found"
    echo "  Recent training output:"
    tail -5 "logs/medium_dense_$SLURM_JOB_ID.log" 2>/dev/null || echo "    No training log found"
else
    echo "  ❌ No medium dense model directory found"
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
echo "  3. Compare metrics across all 4 experiments:"
echo "     - Arena vs Medium maze complexity"
echo "     - Sparse vs Dense reward effectiveness"
echo "     - Environment-reward combination performance"
echo "  4. Test models: python eval_interactive.py (if evaluation script exists)"

echo "========================================================"
echo "🏁 Maze Environment & Reward Comparison Complete - $(date)"