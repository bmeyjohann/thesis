#!/usr/bin/env python3
"""
Unified training script for OGBench environments using RSL-RL PPO.

Features:
- Support for all OGBench environments (pointmaze, antmaze, humanoidmaze)
- Flexible observation configurations
- Interactive configuration when no arguments provided
- Cluster-friendly defaults
- Configurable visualization during training
"""

import os
import sys
import argparse
import torch
import numpy as np
import wandb
from datetime import datetime
from pathlib import Path

# Add RSL-RL to path
sys.path.append('fasttd3/fast_sac')

# Import gymnasium and ogbench
import gymnasium as gym
import ogbench
from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper

# Import RSL-RL components
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class UnifiedOGBenchEnv(VecEnv):
    """Unified RSL-RL VecEnv wrapper for all OGBench environments."""
    
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        
        # Environment configuration
        self.env_name = self.cfg.get('env_name', 'pointmaze-arena-v0')
        self.render_mode = self.cfg.get('render_mode', None)
        self.max_episode_steps = self.cfg.get('max_episode_steps', 500)
        
        # Observation configuration
        self.obs_config = {
            'include_goal': self.cfg.get('include_goal', True),
            'include_distance': self.cfg.get('include_distance', False),
            'include_direction': self.cfg.get('include_direction', False),
            'include_velocity': self.cfg.get('include_velocity', False),
        }
        
        # Reward configuration
        self.reward_config = {
            'reward_type': self.cfg.get('reward_type', 'sparse'),
            'dense_reward_scale': self.cfg.get('dense_reward_scale', 0.01),
            'step_penalty': self.cfg.get('step_penalty', 0.0),
        }
        
        # Create base environment
        env_kwargs = {'render_mode': self.render_mode}
        
        # Add width/height for OGBench environments that support them
        if self.env_name.startswith(('pointmaze-', 'antmaze-', 'humanoidmaze-')):
            env_kwargs.update({
                'width': self.cfg.get('render_width', 800),
                'height': self.cfg.get('render_height', 600),
            })
        
        self.base_env = gym.make(self.env_name, **env_kwargs)
        
        # Apply flexible observation wrapper
        self.base_env = FlexibleObsWrapper(self.base_env, **self.obs_config)
        
        # Apply reward wrapper
        self.base_env = DetailedRewardWrapper(self.base_env, **self.reward_config)
        
        # RSL-RL VecEnv required attributes
        self.num_envs = 1  # Single environment for now
        self.num_actions = self.base_env.action_space.shape[0]
        self.max_episode_length = self.max_episode_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state
        self._last_obs = None
        
        # Initialize observations
        self.reset()
        
    def get_observations(self) -> TensorDict:
        """Return current observations as TensorDict."""
        if self._last_obs is None:
            self.reset()
        return self._last_obs
    
    def reset(self) -> TensorDict:
        """Reset environment and return TensorDict observations."""
        # Reset the environment
        raw_obs, info = self.base_env.reset()
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.from_numpy(raw_obs).unsqueeze(0).to(device=self.device, dtype=torch.float)
        
        # Convert to TensorDict with proper structure for RSL-RL
        self._last_obs = TensorDict({
            "policy": obs_tensor,  # Observations for policy network
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step environment with RSL-RL interface."""
        # Apply action clipping for stability
        processed_actions = torch.clamp(actions, -1.0, 1.0)
        
        # Step the environment
        raw_obs, reward, terminated, truncated, info = self.base_env.step(processed_actions[0].cpu().numpy())
        
        # Convert observations to tensor
        obs_tensor = torch.from_numpy(raw_obs).unsqueeze(0).to(device=self.device, dtype=torch.float)
        
        # Convert to tensors
        rewards = torch.tensor([reward], device=self.device, dtype=torch.float)
        dones = torch.tensor([terminated], device=self.device, dtype=torch.bool)
        
        # Update episode tracking
        self.episode_length_buf += 1
        
        # Track episode statistics and reset on episode end
        reset_mask = terminated or truncated
        episode_length = self.episode_length_buf[0].item() if reset_mask else 0
        
        # Reset episode tracking for terminated episodes
        if reset_mask:
            self.episode_length_buf.zero_()
        
        # Create infos dict with episode statistics
        time_outs = torch.tensor([truncated], device=self.device, dtype=torch.bool)
        extras = {
            'time_outs': time_outs,
            'episode_rewards': rewards.clone(),  # For logging
            'goal_achieved': info.get('success', 0.0),
            'episode_length': episode_length,
            'raw_infos': [info],  # Include raw info for detailed metrics
        }
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[self.num_envs], device=self.device)
        
        self._last_obs = obs_tensordict
        
        return obs_tensordict, rewards, dones, extras


def get_available_environments():
    """Get list of available OGBench environments."""
    return [
        'pointmaze-arena-v0',
        'pointmaze-medium-v0', 
        'pointmaze-large-v0',
        'pointmaze-giant-v0',
        'antmaze-medium-v0',
        'antmaze-large-v0',
        'humanoidmaze-medium-v0',
        'humanoidmaze-large-v0',
    ]


def interactive_config():
    """Interactive configuration when no arguments provided."""
    print("🤖 Interactive Training Configuration")
    print("=" * 50)
    
    config = {}
    
    # Environment selection
    environments = get_available_environments()
    print(f"\nAvailable environments:")
    for i, env in enumerate(environments, 1):
        print(f"   {i}. {env}")
    
    while True:
        try:
            choice = input(f"\nSelect environment (1-{len(environments)}) [1]: ").strip()
            if not choice:
                choice = 1
            else:
                choice = int(choice)
            if 1 <= choice <= len(environments):
                config['env_name'] = environments[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(environments)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Observation configuration
    print(f"\nObservation components:")
    print(f"   Agent position is always included")
    
    config['include_goal'] = input("Include goal position? [Y/n]: ").lower() not in ['n', 'no']
    config['include_distance'] = input("Include distance to goal? [y/N]: ").lower() in ['y', 'yes'] 
    config['include_direction'] = input("Include direction to goal? [y/N]: ").lower() in ['y', 'yes']
    config['include_velocity'] = input("Include velocity? [y/N]: ").lower() in ['y', 'yes']
    
    # Reward configuration
    print(f"\nReward type:")
    print(f"   1. Sparse (goal only)")
    print(f"   2. Dense (distance-based)")
    print(f"   3. Combined (sparse + dense)")
    
    reward_choice = input("Select reward type (1-3) [1]: ").strip() or "1"
    reward_types = {'1': 'sparse', '2': 'dense', '3': 'combined'}
    config['reward_type'] = reward_types.get(reward_choice, 'sparse')
    
    if config['reward_type'] in ['dense', 'combined']:
        config['dense_reward_scale'] = float(input("Dense reward scale [0.01]: ") or 0.01)
    
    # Training configuration
    config['total_timesteps'] = int(input("Total timesteps [1000000]: ") or 1000000)
    config['render_during_training'] = input("Show environment during training? [y/N]: ").lower() in ['y', 'yes']
    
    if config['render_during_training']:
        config['render_mode'] = 'human'
        config['render_interval'] = int(input("Render every N steps [1000]: ") or 1000)
    else:
        config['render_mode'] = None
    
    # Logging
    config['use_wandb'] = input("Use Weights & Biases logging? [y/N]: ").lower() in ['y', 'yes']
    if config['use_wandb']:
        config['wandb_project'] = input("W&B project name [ogbench-training]: ") or 'ogbench-training'
    
    # Experiment name
    config['experiment_name'] = input("Experiment name (leave empty for auto): ").strip()
    
    print(f"\n✅ Configuration complete!")
    return config


def get_args():
    """Parse command line arguments with fallback to interactive config."""
    parser = argparse.ArgumentParser(description='Unified OGBench Training with RSL-RL PPO')
    
    # Environment
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name (e.g., pointmaze-arena-v0)')
    parser.add_argument('--max_episode_steps', type=int, default=None,
                        help='Maximum episode steps')
    
    # Observations
    parser.add_argument('--include_goal', action='store_true', default=None,
                        help='Include goal position in observations')
    parser.add_argument('--include_distance', action='store_true', default=False,
                        help='Include distance to goal')
    parser.add_argument('--include_direction', action='store_true', default=False,
                        help='Include direction to goal')
    parser.add_argument('--include_velocity', action='store_true', default=False,
                        help='Include velocity information')
    
    # Rewards
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['sparse', 'dense', 'combined'],
                        help='Reward type: sparse (goal only), dense (distance), combined')
    parser.add_argument('--dense_reward_scale', type=float, default=0.01,
                        help='Scale factor for dense rewards')
    parser.add_argument('--step_penalty', type=float, default=0.0,
                        help='Small penalty per step')
    
    # Training
    parser.add_argument('--total_timesteps', type=int, default=None,
                        help='Total training timesteps')
    parser.add_argument('--num_steps_per_env', type=int, default=1024,
                        help='Steps per environment per rollout')
    
    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--num_learning_epochs', type=int, default=5,
                        help='Number of learning epochs per update')
    parser.add_argument('--num_mini_batches', type=int, default=4,
                        help='Number of mini-batches per update')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='PPO clip parameter')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--value_loss_coef', type=float, default=1.0,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    
    # Network architecture
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 256, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function')
    
    # Visualization
    parser.add_argument('--render_during_training', action='store_true', default=False,
                        help='Show environment during training')
    parser.add_argument('--render_interval', type=int, default=1000,
                        help='Render every N steps')
    parser.add_argument('--render_width', type=int, default=800,
                        help='Render width')
    parser.add_argument('--render_height', type=int, default=600,
                        help='Render height')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='ogbench-training',
                        help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save model every N iterations')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    # Parse args
    args = parser.parse_args()
    
    # If no environment specified and running interactively, use interactive config
    if args.env_name is None and sys.stdin.isatty():
        print("No environment specified, entering interactive configuration...")
        config = interactive_config()
        
        # Update args with interactive config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        # Set cluster-friendly defaults for unspecified values
        if args.total_timesteps is None:
            args.total_timesteps = config.get('total_timesteps', 1000000)
        if args.include_goal is None:
            args.include_goal = config.get('include_goal', True)
        args.render_mode = config.get('render_mode', None)
    else:
        # Command line mode - set cluster-friendly defaults
        if args.env_name is None:
            args.env_name = 'pointmaze-arena-v0'
        if args.total_timesteps is None:
            args.total_timesteps = 1000000
        if args.include_goal is None:
            args.include_goal = True
        args.render_mode = 'human' if args.render_during_training else None
        
        # Set episode steps based on environment
        if args.max_episode_steps is None:
            if 'arena' in args.env_name:
                args.max_episode_steps = 500
            elif 'medium' in args.env_name:
                args.max_episode_steps = 1000
            else:
                args.max_episode_steps = 1000
    
    return args


def main():
    """Main training function."""
    args = get_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Training RSL-RL PPO on {args.env_name}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_short = args.env_name.replace('-v0', '').replace('maze', '')
        args.experiment_name = f"{env_short}_{timestamp}"
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create environment
    env_cfg = {
        'env_name': args.env_name,
        'max_episode_steps': args.max_episode_steps,
        'render_mode': args.render_mode,
        'render_width': args.render_width,
        'render_height': args.render_height,
        'include_goal': args.include_goal,
        'include_distance': args.include_distance,
        'include_direction': args.include_direction,
        'include_velocity': args.include_velocity,
        'reward_type': args.reward_type,
        'dense_reward_scale': args.dense_reward_scale,
        'step_penalty': args.step_penalty,
    }
    env = UnifiedOGBenchEnv(env_cfg)
    
    print(f"✅ Environment created successfully")
    print(f"   Observation space: {env.base_env.observation_space}")
    print(f"   Action space: {env.base_env.action_space}")
    print(f"   Max episode steps: {env.max_episode_length}")
    
    # Create dummy observation to initialize policy
    dummy_obs = torch.zeros(1, env.get_observations()["policy"].shape[1], device=device)
    dummy_obs_dict = TensorDict({
        "policy": dummy_obs,
    }, batch_size=[1], device=device)
    
    # Create ActorCritic policy
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"]
    }
    
    policy_cfg = {
        'activation': args.activation,
    }
    
    policy = ActorCritic(
        obs=dummy_obs_dict,
        obs_groups=obs_groups,
        num_actions=env.num_actions,
        **policy_cfg
    ).to(device)
    
    print(f"✅ Policy created with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Create PPO algorithm 
    alg_cfg = {
        'value_loss_coef': args.value_loss_coef,
        'use_clipped_value_loss': True,
        'clip_param': args.clip_param,
        'entropy_coef': args.entropy_coef,
        'num_learning_epochs': args.num_learning_epochs,
        'num_mini_batches': args.num_mini_batches,
        'learning_rate': args.learning_rate,
        'schedule': 'adaptive',
        'gamma': args.gamma,
        'lam': args.lam,
        'desired_kl': 0.01,
        'max_grad_norm': 1.0,
    }
    
    ppo = PPO(policy, device=device, **alg_cfg)
    
    # Initialize PPO storage
    dummy_actions = torch.zeros(env.num_envs, env.num_actions, device=device)
    ppo.init_storage(
        "rl",  # training_type for reinforcement learning
        env.num_envs,
        args.num_steps_per_env,
        dummy_obs_dict,  # Use the TensorDict, not the tensor
        dummy_actions.shape
    )
    
    print(f"✅ PPO algorithm initialized")
    
    # Create output directory
    output_dir = Path(f"models/{args.experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    obs = env.reset()
    total_steps = 0
    iteration = 0
    best_reward = -float('inf')
    
    print(f"🏋️  Starting training for {args.total_timesteps:,} total steps...")
    
    while total_steps < args.total_timesteps:
        iteration += 1
        
        # Collect rollout with detailed metrics
        episode_rewards = []
        episode_lengths = []
        sparse_rewards = []
        dense_rewards = []
        goals_reached = []
        distances = []
        
        for step in range(args.num_steps_per_env):
            with torch.no_grad():
                actions = ppo.act(obs)
            
            obs, rewards, dones, extras = env.step(actions)
            
            ppo.process_env_step(obs, rewards, dones, extras)
            total_steps += env.num_envs
            
            # Track detailed episode statistics
            if 'episode_rewards' in extras:
                episode_rewards.append(extras['episode_rewards'].item())
            if 'episode_length' in extras and extras['episode_length'] > 0:
                episode_lengths.append(extras['episode_length'])
                
                # Collect detailed metrics from the raw_infos if available
                raw_infos = extras.get('raw_infos', [{}])
                if raw_infos and len(raw_infos) > 0:
                    info = raw_infos[0] if isinstance(raw_infos, list) else raw_infos
                    sparse_rewards.append(info.get('episode_sparse_reward', 0))
                    dense_rewards.append(info.get('episode_dense_reward', 0))
                    goals_reached.append(info.get('goal_reached', False))
                    distances.append(info.get('distance_to_goal', 0))
            
            # Render if requested
            if args.render_during_training and total_steps % args.render_interval == 0:
                env.base_env.render()
            
            if total_steps >= args.total_timesteps:
                break
        
        # Learning step
        ppo.compute_returns(obs)
        loss_dict = ppo.update()
        
        # Extract losses
        mean_value_loss = loss_dict.get('value_loss', 0.0)
        mean_surrogate_loss = loss_dict.get('surrogate_loss', 0.0)
        mean_kl_div = loss_dict.get('kl', 0.0)
        
        # Detailed logging and checkpointing
        if episode_rewards:
            # Calculate basic metrics
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths) if episode_lengths else 0
            
            # Calculate detailed metrics
            mean_sparse_reward = np.mean(sparse_rewards) if sparse_rewards else 0
            mean_dense_reward = np.mean(dense_rewards) if dense_rewards else 0
            goal_success_rate = np.mean(goals_reached) if goals_reached else 0
            mean_distance = np.mean(distances) if distances else 0
            
            # Update best reward and save model
            if mean_reward > best_reward:
                best_reward = mean_reward
                
            # Save checkpoint every iteration (not just best)
            checkpoint = {
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
                'total_timesteps': total_steps,
                'best_reward': best_reward,
                'mean_reward': mean_reward,
                'goal_success_rate': goal_success_rate,
                'args': args,
                'training_metrics': {
                    'sparse_reward': mean_sparse_reward,
                    'dense_reward': mean_dense_reward,
                    'total_reward': mean_reward,
                    'episode_length': mean_length,
                    'goal_success_rate': goal_success_rate,
                    'distance_to_goal': mean_distance,
                }
            }
            
            # Save both current and best model
            torch.save(checkpoint, output_dir / f'checkpoint_iter_{iteration:04d}.pt')
            if mean_reward >= best_reward:
                torch.save(checkpoint, output_dir / 'best_model.pt')
            
            print(f"Iter {iteration:4d} | Steps: {total_steps:7,} | "
                  f"Reward: {mean_reward:7.3f} | Sparse: {mean_sparse_reward:6.3f} | Dense: {mean_dense_reward:7.4f} | "
                  f"Success: {goal_success_rate:5.1%} | Length: {mean_length:5.1f}")
            print(f"           | VLoss: {mean_value_loss:.4f} | PLoss: {mean_surrogate_loss:.4f} | "
                  f"Distance: {mean_distance:6.2f}")
            
            # Create log data for both wandb and CSV
            log_data = {
                'iteration': iteration,
                'total_timesteps': total_steps,
                'episode_reward_total': mean_reward,
                'episode_reward_sparse': mean_sparse_reward,
                'episode_reward_dense': mean_dense_reward,
                'episode_length_mean': mean_length,
                'goal_success_rate': goal_success_rate,
                'distance_to_goal': mean_distance,
                'value_loss': mean_value_loss,
                'policy_loss': mean_surrogate_loss,
                'kl_divergence': mean_kl_div,
                'best_reward': best_reward,
                'reward_type': args.reward_type,
            }
            
            # Wandb logging (if enabled)
            if args.use_wandb:
                wandb.log(log_data)
                
            # Always log to CSV (offline-friendly)
            import csv
            csv_file = output_dir / 'training_log.csv'
            file_exists = csv_file.exists()
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)
        
        # Additional checkpoint saving (already done above, but keep for backwards compatibility)
    
    # Save final model
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'iteration': iteration,
        'total_timesteps': total_steps,
        'best_reward': best_reward,
        'args': args,
    }
    torch.save(checkpoint, output_dir / 'final_model.pt')
    
    print(f"✅ Training complete!")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Best reward: {best_reward:.3f}")
    print(f"   Models saved to: {output_dir}")
    
    env.base_env.close()
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()