# JUWELS Cluster Setup Instructions

## 🚀 First-time Setup

### 1. Upload Code to Cluster
```bash
# From your local machine
rsync -avz --exclude=.git --exclude=wandb --exclude=models \
    /path/to/thesis/ <username>@juwels-booster.fz-juelich.de:~/thesis/
```

### 2. Create Virtual Environment (One-time setup)
```bash
# On JUWELS login node
cd ~/thesis/sc_venv_template
bash setup.sh  # Creates venv and installs all packages from requirements.txt
```

### 3. Test Interactive Session
```bash
# Request interactive node
salloc --account=<your_account> \
       --nodes=1 \
       --cpus-per-task=24 \
       --gres=gpu:1 \
       --time=01:00:00 \
       --partition=booster

# SSH to allocated node (e.g. jrc0123)
ssh jrc0123

# Navigate and activate environment
cd ~/thesis
source sc_venv_template/activate.sh

# Test training
python train_rsl_rl_clean.py --num_envs 2 --max_iterations 1 --reward_type dense
```

### 4. Setup Wandb (One-time)
```bash
# IMPORTANT: Only do this on login node (has internet access)
cd ~/thesis
source sc_venv_template/activate.sh
wandb login  # Follow prompts with your API key
```

**Note**: Compute nodes have **no internet access**, so wandb will run in offline mode during training and sync automatically after jobs complete.

## 🎯 Production Training

### Submit Pilot Experiments
```bash
# Update account in script first
vim juwels_reward_comparison.sbatch
# Change: #SBATCH --account=<your_account>

# Submit job
sbatch juwels_reward_comparison.sbatch

# Monitor
squeue -u $USER
```

### Monitor Progress
```bash
# Check job output (live monitoring)
tail -f logs/reward_comparison_*.out

# Check CSV logs (backup, always works)
tail -f logs/*_metrics.csv

# Sync wandb runs (after job completes)
bash sync_wandb.sh

# Check wandb dashboard (after syncing)
# URL: https://wandb.ai/<username>/juwels-reward-pilot
```

## 📊 **Logging Strategy (No Internet on Compute Nodes)**

| Method | Location | When | Pros | Cons |
|--------|----------|------|------|------|
| **Wandb Offline** | Compute node | During training | Full wandb features | Need to sync later |
| **CSV Backup** | `logs/*.csv` | During training | Always works | Basic metrics only |
| **Console Output** | `logs/*.out` | During training | Real-time | Text only |
| **Wandb Online** | Login node | After syncing | Full dashboard | Post-training only |

## 📦 What's Included in Virtual Environment

The `sc_venv_template/requirements.txt` already includes:
- ✅ PyTorch 2.6.0 + CUDA support
- ✅ Wandb 0.21.0 for logging  
- ✅ RSL-RL 3.0.0 for PPO
- ✅ Gymnasium 1.2.0 for environments
- ✅ TensorDict 0.7.2 for observations
- ✅ Stable-Baselines3 2.6.0 for vectorized envs
- ✅ OGBench (editable install)

## 🎛️ Cluster-Specific Notes

- **No conda**: Uses Python virtual environments with system modules
- **Module system**: Automatically loads optimized PyTorch, GCC, etc.
- **GPU support**: CUDA libraries pre-installed via modules  
- **MPI support**: Multi-node training ready (if needed later)

## 🚨 Common Issues

| Issue | Solution |
|-------|---------|
| "Module not found" | Run `source sc_venv_template/activate.sh` first |
| "GPU not available" | Check `--gres=gpu:N` in SLURM script |
| "Permission denied" | Verify account: `sacctmgr show assoc user=$USER` |
| "Wandb offline" | Run `wandb login` in interactive session |
