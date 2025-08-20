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
from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, VectorizedOGBenchEnv

# Import RSL-RL components
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict



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
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments')
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
    
    # Initialize wandb if requested (with error handling for cluster environments)
    wandb_active = False
    if args.use_wandb:
        # Aggressive offline mode for compute nodes - prevent all network attempts
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_CONSOLE"] = "off"  # Reduce verbose output
        os.environ["WANDB_SILENT"] = "true"  # Suppress network retry messages
        
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args),
                mode="offline"
            )
            wandb_active = True
            print("✅ Wandb initialized successfully")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
            print("📄 Continuing with CSV logging only...")
            wandb_active = False
    
    # Create wrapper functions for environment pipeline
    def apply_wrappers(env):
        # Apply flexible observation wrapper
        env = FlexibleObsWrapper(
            env,
            include_goal=args.include_goal,
            include_distance=args.include_distance,
            include_direction=args.include_direction,
            include_velocity=args.include_velocity,
        )
        
        # Apply detailed reward wrapper
        env = DetailedRewardWrapper(
            env,
            reward_type=args.reward_type,
            dense_reward_scale=args.dense_reward_scale,
            step_penalty=args.step_penalty,
        )
        
        return env
    
    # Create vectorized environment
    env = VectorizedOGBenchEnv(
        env_name=args.env_name,
        num_envs=args.num_envs,
        wrappers=[apply_wrappers],
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
    )
    
    print(f"✅ Environment created successfully")
    print(f"   Number of environments: {env.num_envs}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Max episode steps: {env.max_episode_length}")
    
    # Create dummy observation to initialize policy
    # Reset environment to get observation shape
    initial_obs = env.reset()
    obs_shape = initial_obs["policy"].shape[1]  # Get feature dimension
    
    dummy_obs = torch.zeros(env.num_envs, obs_shape, device=device)
    dummy_obs_dict = TensorDict({
        "policy": dummy_obs,
    }, batch_size=[env.num_envs], device=device)
    
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
    action_shape = (env.num_actions,)  # Shape tuple for actions
    ppo.init_storage(
        "rl",  # training_type for reinforcement learning
        env.num_envs,
        args.num_steps_per_env,
        dummy_obs_dict,  # Use the TensorDict, not the tensor
        action_shape
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
            
            # Track detailed episode statistics (handle multiple environments)
            if 'episode_rewards' in extras:
                # episode_rewards is now a list (one per completed episode)
                if isinstance(extras['episode_rewards'], list):
                    episode_rewards.extend(extras['episode_rewards'])
                else:
                    episode_rewards.append(extras['episode_rewards'])
                    
            if 'episode_lengths' in extras:
                # episode_lengths is now a list (one per completed episode)
                if isinstance(extras['episode_lengths'], list):
                    episode_lengths.extend(extras['episode_lengths'])
                else:
                    episode_lengths.append(extras['episode_lengths'])
            
            # Collect detailed metrics from completed episodes
            if 'episode_sparse_rewards' in extras:
                sparse_rewards.extend(extras['episode_sparse_rewards'])
            if 'episode_dense_rewards' in extras:
                dense_rewards.extend(extras['episode_dense_rewards'])
            if 'goals_reached' in extras:
                goals_reached.extend(extras['goals_reached'])
            if 'distances_to_goal' in extras:
                distances.extend(extras['distances_to_goal'])
            
            # Render if requested (only works with single environment)
            if args.render_during_training and total_steps % args.render_interval == 0:
                if env.num_envs == 1 and hasattr(env, 'envs') and len(env.envs) > 0:
                    env.envs[0].render()
            
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
            
            # Calculate detailed metrics (works for any number of environments)
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
            
            # Show number of completed episodes for monitoring parallel training
            num_episodes = len(episode_rewards)
            # Calculate mean reward per step for additional context
            mean_reward_per_step = mean_reward / mean_length if mean_length > 0 else 0
            
            print(f"Iter {iteration:4d} | Steps: {total_steps:7,} | Episodes: {num_episodes:3d} | "
                  f"Reward: {mean_reward:7.3f} | MeanPerStep: {mean_reward_per_step:7.4f}")
            print(f"           | Sparse: {mean_sparse_reward:6.3f} | Dense: {mean_dense_reward:7.4f} | "
                  f"Success: {goal_success_rate:5.1%} | Length: {mean_length:5.1f}")
            print(f"           | VLoss: {mean_value_loss:.4f} | PLoss: {mean_surrogate_loss:.4f} | Distance: {mean_distance:6.2f}")
            
            # Create log data for both wandb and CSV
            log_data = {
                'iteration': iteration,
                'total_timesteps': total_steps,
                'episodes_completed': num_episodes,
                'episode_reward_total': mean_reward,
                'episode_reward_per_step': mean_reward_per_step,
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
            
            # Wandb logging (if enabled and working)
            if args.use_wandb and wandb_active:
                try:
                    wandb.log(log_data)
                except Exception as e:
                    print(f"⚠️  Wandb logging failed: {e}")
                    wandb_active = False
                
            # Always log to CSV (offline-friendly)
            import csv
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)
            csv_file = logs_dir / f'{args.experiment_name}_training.csv'
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
    
    env.close()
    if args.use_wandb and wandb_active:
        try:
            wandb.finish()
        except Exception as e:
            print(f"⚠️  Wandb finish failed: {e}")


if __name__ == "__main__":
    main()