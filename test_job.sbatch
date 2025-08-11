#!/bin/bash
#SBATCH --account=<your_account>  # This will be auto-replaced by submit_job.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=booster
#SBATCH --job-name=ogbench_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Create logs directory
mkdir -p logs

# Activate environment
cd sc_venv_template
source activate.sh
cd ..

echo "🧪 Testing OGBench RSL-RL Setup on JUWELS"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L)"
date

# Quick test run
echo "🔬 Running minimal test (2 envs, 1 iteration)..."
python train_rsl_rl_clean.py \
    --num_envs 2 \
    --max_iterations 1 \
    --reward_type dense \
    --experiment_name "test_juwels_$(date +%Y%m%d_%H%M)" \
    --use_wandb

echo "✅ Test completed successfully!"
echo "Check the output above for any errors."
echo "If successful, submit the main job with: ./submit_job.sh"