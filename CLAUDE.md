# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contributor Guide

For coding standards, project layout, and workflow, see `AGENTS.md` (Repository Guidelines).

## Project Overview

This is a reinforcement learning research repository focused on training agents on OGBench environments using various algorithms including RSL-RL PPO and FastTD3. The project includes implementations for both local development and HPC cluster deployment (JUWELS supercomputer).

## Key Training Scripts

### Local Development
- `train_rsl_rl_clean.py` - Main RSL-RL PPO training script with proper TensorDict integration
- `train.py` - Alternative RSL-RL training implementation
- `train_pointmaze*.py` - Specialized point maze training variants (parallel, dynamic, visual)

### Cluster Training
- `submit_job.sh` - Smart SLURM job submission with automatic account detection
- `test_job.sbatch` - Quick test job for JUWELS cluster validation
- `juwels_reward_comparison.sbatch` - Main experiment job template

### Environment Setup
- `sc_venv_template/` - HPC virtual environment template for JUWELS
- `fasttd3/requirements/requirements.txt` - FastTD3 dependencies
- `ogbench/impls/requirements.txt` - OGBench algorithm dependencies

## Common Commands

### Local Training
```bash
# Basic RSL-RL training
python train_rsl_rl_clean.py --env_name pointmaze-medium-v0 --use_wandb

# FastTD3 training
cd fasttd3 && python fast_sac/train.py

# Interactive evaluation
python eval_interactive.py
```

### Cluster Operations
```bash
# Submit main experiment
./submit_job.sh

# Submit test job
sbatch test_job.sbatch

# Monitor job progress
tail -f logs/reward_comparison_*.out
squeue -u $USER

# Sync wandb after completion
bash sync_wandb.sh
```

### Environment Management
```bash
# Activate HPC environment
cd sc_venv_template && source activate.sh

# Activate FastTD3 environment (local development)
conda activate fasttd3

# Install OGBench in development mode
pip install -e ogbench/

# Install FastTD3
pip install -e fasttd3/
```

## Architecture

### Environment Wrappers
- `ogbench/ogbench/wrappers/enhanced_obs_wrapper.py` - Enhanced observation wrappers for goal-conditioned navigation
- `ogbench/ogbench/wrappers/goal_conditioned_wrapper.py` - Goal-conditioned observation wrapper
- Compatible with RSL-RL, FastTD3, and other algorithms through standard gym interfaces

### Key Components
- **OGBench Integration**: Point maze, ant maze, and humanoid maze environments
- **RSL-RL PPO**: Actor-critic implementation with proper TensorDict observations
- **FastTD3**: Alternative SAC-based algorithm implementation
- **Parallel Training**: SubprocVecEnv for efficient multi-environment training

### Logging Strategy
- **Wandb**: Primary logging (offline mode on cluster compute nodes)
- **CSV Backup**: `logs/*.csv` for reliable metric tracking
- **Console Output**: Real-time monitoring via SLURM output files

## HPC Cluster (JUWELS) Specifics

### Module System
The cluster uses environment modules instead of conda:
- PyTorch and CUDA via system modules
- Python virtual environments in `sc_venv_template/`
- MPI support available for multi-node training

### Account Management
The `submit_job.sh` script automatically detects SLURM accounts using multiple methods:
1. Saved config file (`.slurm_account`)
2. Environment variable (`SLURM_ACCOUNT`)
3. `sacctmgr` queries
4. Interactive prompt as fallback

### Offline Wandb Workflow
1. Training runs in offline mode on compute nodes
2. Use `sync_wandb.sh` to upload runs from login nodes
3. CSV logs provide backup metrics that always work

## Important Notes

- Use standard `gym.make()` with OGBench environment IDs (e.g. 'pointmaze-arena-v0')
- Apply OGBench wrappers as needed for enhanced observations
- RSL-RL requires TensorDict observations - use custom VecEnv wrapper
- For cluster jobs, replace `<your_account>` placeholders or use `submit_job.sh`
- FastTD3 requires specific PyTorch/CUDA versions (see requirements)
- The repository includes both individual OGBench and FastTD3 implementations as submodules

## Goal Navigation Experiments

### Quick Cluster Training
```bash
# Submit the four-experiment comparison
./submit_job.sh goal_navigation_experiments.sbatch

# Monitor progress
tail -f logs/goal_nav_experiments_*.out
```

### Experiment Configurations
1. **Simple + Sparse**: SimpleDynamicPointMaze-v0 with goal-only rewards
2. **Simple + Dense**: SimpleDynamicPointMaze-v0 with distance-based rewards  
3. **OGBench + Sparse**: pointmaze-medium-v0 with goal-only rewards
4. **OGBench + Dense**: pointmaze-medium-v0 with distance-based rewards

All experiments use enhanced observations with goal position, distance, and direction information.
