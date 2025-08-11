#!/bin/bash
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00  # 10 minutes
#SBATCH --partition=booster
#SBATCH --job-name=ogbench_test

module load Stages/2024
module load GCC/11.3.0 CUDA/11.7 Python/3.10.4
source activate fasttd3

# Test run - very short
python train_rsl_rl_clean.py --num_envs 2 --max_iterations 1 --reward_type dense