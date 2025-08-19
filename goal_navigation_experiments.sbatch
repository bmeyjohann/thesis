#!/bin/bash
#SBATCH --account=<your_account>  # This will be auto-replaced by submit_job.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --job-name=goal_navigation_experiments
#SBATCH --output=logs/goal_nav_experiments_%j.out
#SBATCH --error=logs/goal_nav_experiments_%j.err

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

echo "🧪 Starting Goal Navigation Experiments on JUWELS"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
date

# Define shared training parameters
# Adjusted for longer training to see clear learning differences
SHARED_ARGS="--num_envs 64 --total_timesteps 20000000 --num_steps_per_env 512 --log_interval 20"
WANDB_ARGS="--wandb_project goal-navigation-juwels --use_wandb"

# Generate timestamp for experiment naming
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "🎯 Launching 4 parallel goal navigation experiments..."
echo "📊 Experimental Design:"
echo "  1. Simple + Sparse:  SimpleDynamicPointMaze with goal-only rewards"
echo "  2. Simple + Dense:   SimpleDynamicPointMaze with distance-based rewards"  
echo "  3. OGBench + Sparse: pointmaze-medium-v0 with goal-only rewards"
echo "  4. OGBench + Dense:  pointmaze-medium-v0 with distance-based rewards"
echo ""

# Experiment 1: Simple Dynamic Point Maze + Sparse Rewards
echo "🔵 Starting Experiment 1: Simple + Sparse..."
python train_goal_navigation_experiments.py \
    --env_type simple \
    --reward_type sparse \
    --experiment_name "simple_sparse_${TIMESTAMP}" \
    $SHARED_ARGS $WANDB_ARGS &
PID1=$!

# Experiment 2: Simple Dynamic Point Maze + Dense Rewards  
echo "🟢 Starting Experiment 2: Simple + Dense..."
python train_goal_navigation_experiments.py \
    --env_type simple \
    --reward_type dense \
    --experiment_name "simple_dense_${TIMESTAMP}" \
    $SHARED_ARGS $WANDB_ARGS &
PID2=$!

# Experiment 3: OGBench Point Maze + Sparse Rewards
echo "🔴 Starting Experiment 3: OGBench + Sparse..."
python train_goal_navigation_experiments.py \
    --env_type ogbench \
    --reward_type sparse \
    --experiment_name "ogbench_sparse_${TIMESTAMP}" \
    $SHARED_ARGS $WANDB_ARGS &
PID3=$!

# Experiment 4: OGBench Point Maze + Dense Rewards
echo "🟡 Starting Experiment 4: OGBench + Dense..."
python train_goal_navigation_experiments.py \
    --env_type ogbench \
    --reward_type dense \
    --experiment_name "ogbench_dense_${TIMESTAMP}" \
    $SHARED_ARGS $WANDB_ARGS &
PID4=$!

echo "✓ All 4 experiments launched in parallel"
echo "Process IDs: Simple_Sparse=$PID1, Simple_Dense=$PID2, OGBench_Sparse=$PID3, OGBench_Dense=$PID4"
echo "Monitor progress at: https://wandb.ai/<your_username>/goal-navigation-juwels"

# Wait for all jobs to complete with error checking
echo "⏳ Waiting for all 4 experiments to complete..."

wait $PID1
if [ $? -eq 0 ]; then
    echo "✅ Simple + Sparse experiment completed successfully"
else
    echo "❌ Simple + Sparse experiment failed (exit code: $?)"
fi

wait $PID2  
if [ $? -eq 0 ]; then
    echo "✅ Simple + Dense experiment completed successfully"
else
    echo "❌ Simple + Dense experiment failed (exit code: $?)"
fi

wait $PID3
if [ $? -eq 0 ]; then
    echo "✅ OGBench + Sparse experiment completed successfully"
else
    echo "❌ OGBench + Sparse experiment failed (exit code: $?)"
fi

wait $PID4
if [ $? -eq 0 ]; then
    echo "✅ OGBench + Dense experiment completed successfully"
else
    echo "❌ OGBench + Dense experiment failed (exit code: $?)"
fi

echo "🎉 All goal navigation experiments completed!"
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

# Generate final summary
echo ""
echo "📈 Goal Navigation Experiments Summary:"
echo "========================================"
echo ""
echo "🔬 Experimental Conditions:"
echo "  Simple + Sparse:    SimpleDynamicPointMaze-v0 with goal-only rewards (baseline)"
echo "  Simple + Dense:     SimpleDynamicPointMaze-v0 with distance-based rewards"
echo "  OGBench + Sparse:   pointmaze-medium-v0 with goal-only rewards"
echo "  OGBench + Dense:    pointmaze-medium-v0 with distance-based rewards"
echo ""
echo "⚙️  Training Configuration:"
echo "  - Environments: 64 parallel per experiment"
echo "  - Timesteps: 20M per experiment (80M total)"
echo "  - Steps per update: 512"
echo "  - Algorithm: PPO with RSL-RL"
echo "  - Observations: Enhanced with goal position, distance, and direction"
echo ""
echo "🕐 Expected Results:"
echo "  - Simple + Dense should learn fastest (open space + dense rewards)"
echo "  - OGBench + Dense should learn well but slower (obstacles + dense rewards)"
echo "  - Simple + Sparse should learn slowly (open space but sparse rewards)" 
echo "  - OGBench + Sparse should be most challenging (obstacles + sparse rewards)"
echo ""
echo "📊 Results available in:"
echo "  - Wandb dashboard: https://wandb.ai/<username>/goal-navigation-juwels"
echo "  - Model files: ./models/simple_*_${TIMESTAMP}/ and ./models/ogbench_*_${TIMESTAMP}/"
echo "  - Logs: ./logs/goal_nav_experiments_${SLURM_JOB_ID}.out"
echo ""
echo "🎯 Key Metrics to Compare:"
echo "  - Sample efficiency (steps to reach consistent goal achievement)"
echo "  - Final success rate and episode rewards"
echo "  - Learning curve stability"
echo "  - Policy exploration patterns"
echo ""
echo "💡 Next Steps:"
echo "  1. Analyze learning curves in Wandb"
echo "  2. Compare sample efficiency across conditions"
echo "  3. Evaluate trained policies with: python eval_interactive.py"
echo "  4. Run additional seeds if results are promising"

echo ""
echo "🏁 Experiment batch completed at $(date)"