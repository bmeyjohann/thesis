# OGBench RSL-RL PPO Training

This repository provides a complete implementation for training RL agents on OGBench point locomotion environments using RSL-RL's PPO algorithm, with a reusable environment wrapper that's also compatible with FastTD3.

## Overview

- **Environment Wrapper**: `fasttd3/fast_sac/environments/ogbench_env.py` - A gymnasium-style wrapper for OGBench environments that follows the same pattern as other environment wrappers in the codebase
- **Training Script**: `train.py` - RSL-RL PPO training script for point locomotion environments
- **Compatibility**: The environment wrapper is designed to work with both RSL-RL and FastTD3 algorithms

## Environment Wrapper (`ogbench_env.py`)

The `OGBenchEnv` class wraps OGBench environments to provide:

- **Parallel Environment Support**: Uses SubprocVecEnv for efficient parallel training
- **PyTorch Integration**: Automatic tensor conversion and device management
- **Compatibility**: Follows the same interface pattern as other environment wrappers
- **Flexibility**: Works with any OGBench environment (pointmaze, antmaze, humanoidmaze, etc.)

### Key Features

```python
from fasttd3.fast_sac.environments.ogbench_env import OGBenchEnv

# Create environment with parallel workers
env = OGBenchEnv(
    env_name='pointmaze-medium-v0',
    num_envs=256,  # Number of parallel environments
    device='cuda'
)

# Standard gymnasium-style interface
obs = env.reset()  # Returns torch.Tensor
obs, rewards, dones, infos = env.step(actions)  # All tensors
```

## Training Script (`train.py`)

The training script provides a complete RSL-RL PPO implementation for OGBench environments.

### Usage

```bash
# Basic training
python train.py --env_name pointmaze-medium-v0 --num_envs 256

# With wandb logging
python train.py --env_name pointmaze-medium-v0 --num_envs 256 --use_wandb

# Custom hyperparameters
python train.py \
    --env_name pointmaze-large-v0 \
    --num_envs 512 \
    --total_timesteps 2000000 \
    --learning_rate 3e-4 \
    --hidden_dims 512,512 \
    --use_wandb
```

### Key Arguments

- `--env_name`: OGBench environment name (e.g., 'pointmaze-medium-v0', 'pointmaze-large-v0')
- `--num_envs`: Number of parallel environments (default: 256)
- `--total_timesteps`: Total training timesteps (default: 1000000)
- `--learning_rate`: Learning rate for policy and value networks (default: 3e-4)
- `--hidden_dims`: Network architecture (default: '256,256')
- `--use_wandb`: Enable Weights & Biases logging

### Available Environments

The script works with all OGBench point locomotion environments:

- `pointmaze-medium-v0` - Medium-sized maze
- `pointmaze-large-v0` - Large maze
- `pointmaze-giant-v0` - Giant maze
- `pointmaze-teleport-v0` - Teleportation maze

## Installation & Setup

1. **Activate the environment**:
   ```bash
   conda activate fasttd3
   ```

2. **Verify dependencies**:
   - `ogbench` - OGBench environments
   - `rsl_rl` - RSL-RL algorithms
   - `torch` - PyTorch
   - `stable_baselines3` - For SubprocVecEnv
   - `wandb` - For logging (optional)

3. **Run training**:
   ```bash
   python train.py --env_name pointmaze-medium-v0
   ```

## Compatibility with FastTD3

The `OGBenchEnv` wrapper is designed to be compatible with FastTD3 algorithms. It follows the same interface pattern as other environment wrappers in the `fasttd3/fast_sac/environments/` directory:

```python
# Use with FastTD3 (similar to existing training scripts)
from fasttd3.fast_sac.environments.ogbench_env import OGBenchEnv

env = OGBenchEnv('pointmaze-medium-v0', num_envs=1024)

# Standard interface compatible with FastTD3 training loops
obs = env.reset()
obs, rewards, dones, infos = env.step(actions)
```

## Output

- **Models**: Saved to `models/{experiment_name}/`
- **Logs**: Console output + optional wandb logging
- **Checkpoints**: Periodic model saves during training

## Example Training Output

```
Using device: cuda
Creating environment: pointmaze-medium-v0
Environment created with 256 parallel environments
Observation space: 2
Action space: 2
Starting training...
Iteration     10 | Timesteps:    25600 | Value Loss: 0.1234 | Policy Loss: 0.0567 | Entropy: 0.0123 | FPS: 4096
Iteration     20 | Timesteps:    51200 | Value Loss: 0.0987 | Policy Loss: 0.0432 | Entropy: 0.0098 | FPS: 4152
...
Evaluation | Mean Reward: 0.85 ± 0.12 | Mean Length: 234.5 ± 45.2
```

## File Structure

```
thesis/
├── train.py                           # RSL-RL PPO training script
├── fasttd3/fast_sac/environments/
│   ├── ogbench_env.py                 # OGBench environment wrapper
│   ├── humanoid_bench_env.py          # Existing wrapper
│   ├── isaaclab_env.py                # Existing wrapper
│   └── mujoco_playground_env.py       # Existing wrapper
├── ogbench/                           # OGBench repository
└── models/                            # Generated model checkpoints
```

This implementation provides a clean, reusable solution for training RL agents on OGBench environments that can easily be extended to other algorithms like FastTD3.


