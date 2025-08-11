
# Or edit script manually
vim juwels_reward_comparison.sbatch
# Change: #SBATCH --account=<your_account>
sbatch juwels_reward_comparison.sbatch
```

**Option C: Config File Method**
```bash
# Save account permanently
echo 'export SLURM_ACCOUNT="your_actual_account"' > .slurm_account
./submit_job.sh  # Will use saved account automatically
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
