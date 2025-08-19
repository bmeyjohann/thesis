#!/usr/bin/env python3
"""
Goal Navigation Experiments Training Script  
==========================================

Based on the working train_rsl_rl_clean.py script structure.
Compares goal navigation learning across:
- Simple Dynamic Point Maze vs OGBench Point Maze environments  
- Sparse vs Dense reward structures
- Enhanced observations with goal information

Usage:
    python train_goal_navigation_experiments.py \
        --env_type simple \
        --reward_type dense \
        --experiment_name exp1
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import wandb

# Add the fasttd3 directory to path for imports
sys.path.append('fasttd3/fast_sac')

# Import RSL-RL components
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

# Import ogbench and register environments
import ogbench
import gymnasium as gym

# Import our environment wrappers
from environments.ogbench_env import OGBenchEnv, SimpleDynamicRSLRLVecEnv, OGBenchRSLRLVecEnv, _log_to_csv

# Register the simple environment
gym.register(
    id='SimpleDynamicPointMaze-v0',
    entry_point='simple_dynamic_pointmaze:SimpleDynamicPointMaze',
    max_episode_steps=500,
)


# Environment wrappers are now imported from goal_navigation_env.py


def main():
    parser = argparse.ArgumentParser(description='Goal Navigation Experiments')
    
    # Environment settings
    parser.add_argument('--env_type', type=str, choices=['simple', 'ogbench'], required=True,
                        help='Environment type: simple (SimpleDynamicPointMaze) or ogbench (pointmaze-medium-v0)')
    parser.add_argument('--reward_type', type=str, choices=['sparse', 'dense'], required=True,
                        help='Reward type: sparse (goal only) or dense (distance-based)')
    
    # Training settings  
    parser.add_argument('--num_envs', type=int, default=32,
                        help='Number of parallel environments (only for ogbench)')
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                        help='Total training timesteps')
    parser.add_argument('--num_steps_per_env', type=int, default=512,
                        help='Steps collected before each policy update')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='Maximum training iterations')
    
    # PPO settings
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--num_learning_epochs', type=int, default=5,
                        help='Number of learning epochs per update')
    parser.add_argument('--num_mini_batches', type=int, default=4,
                        help='Number of mini-batches per update')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--value_loss_coef', type=float, default=1.0,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Architecture
    parser.add_argument('--hidden_dims', type=str, default='256,256',
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function')
    
    # Logging settings
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name for logging and model saving')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='Log every N iterations')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='goal-navigation-experiments',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Goal Navigation Experiment: {args.experiment_name}")
    print(f"   Environment: {args.env_type}")
    print(f"   Reward Type: {args.reward_type}")
    print(f"   Device: {device}")
    
    # Initialize wandb if requested
    wandb_initialized = False
    if args.use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )
            wandb_initialized = True
            print(f"✓ Wandb initialized: {args.wandb_project}")
        except Exception as e:
            print(f"❌ Wandb initialization failed: {e}")
            print("🔄 Continuing training without wandb logging...")
    
    # Create environment
    print(f"\n🏗️  Creating environment...")
    try:
        if args.env_type == 'simple':
            # Create simple dynamic environment
            env_cfg = {
                'reward_type': args.reward_type,
                'arena_size': 20.0,
                'max_episode_steps': 500,
                'action_scale': 0.5,
                'goal_threshold': 0.5,
            }
            env = SimpleDynamicRSLRLVecEnv(cfg=env_cfg)
            obs_dim = 7  # Enhanced observations: pos(2) + goal(2) + distance(1) + direction(2)
            
        else:  # ogbench
            # Enhanced observation configuration for OGBench
            obs_config = {
                'include_goal': True,
                'include_goal_distance': True, 
                'include_velocity': False,  # Not always available
                'include_goal_direction': True,
                'normalize_positions': False,
            }
            
            base_env = OGBenchEnv(
                env_name='pointmaze-medium-v0',
                num_envs=args.num_envs,
                obs_config=obs_config
            )
            
            env_cfg = {
                'reward_type': args.reward_type,
                'dense_reward_scale': 0.01,
            }
            env = OGBenchRSLRLVecEnv(base_env, cfg=env_cfg)
            obs_dim = base_env.num_obs
        
        print(f"✓ Environment created successfully")
        print(f"  - Environments: {env.num_envs}")
        print(f"  - Observations: {obs_dim}")
        print(f"  - Actions: {env.num_actions}")
        print(f"  - Max episode length: {env.max_episode_length}")
        print(f"  - Reward type: {args.reward_type}")
        
        # Test TensorDict structure
        test_obs = env.get_observations()
        print(f"  - TensorDict keys: {list(test_obs.keys())}")
        print(f"  - Policy obs shape: {test_obs['policy'].shape}")
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create policy configuration
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    policy_cfg = {
        'init_noise_std': 1.0,
        'actor_hidden_dims': hidden_dims,
        'critic_hidden_dims': hidden_dims,
        'activation': args.activation,
    }
    
    # Create ActorCritic policy
    print(f"\n🧠 Creating ActorCritic policy...")
    try:
        # Get dummy observations as TensorDict
        dummy_obs = env.get_observations()
        
        # RSL-RL observation groups - map to TensorDict keys
        obs_groups = {
            "policy": ["policy"],  # Actor observations
            "critic": ["policy"]   # Critic observations (same as policy for symmetric obs)
        }
        
        print(f"  - Obs groups: {obs_groups}")
        print(f"  - Policy obs shape: {dummy_obs['policy'].shape}")
        
        # Create policy on CPU first to avoid CUDA issues
        dummy_obs_cpu = dummy_obs.to("cpu")
        
        policy = ActorCritic(
            obs=dummy_obs_cpu,
            obs_groups=obs_groups,
            num_actions=env.num_actions,
            **policy_cfg
        ).to(device)
        
        print(f"✓ ActorCritic created successfully")
        print(f"  - Policy device: {next(policy.parameters()).device}")
        
    except Exception as e:
        print(f"❌ ActorCritic creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create PPO algorithm
    print(f"\n⚙️  Creating PPO algorithm...")
    try:
        ppo = PPO(
            policy=policy,
            num_learning_epochs=args.num_learning_epochs,
            num_mini_batches=args.num_mini_batches,
            clip_param=args.clip_param,
            gamma=args.gamma,
            lam=args.lam,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True,
            device=device,
        )
        
        print(f"✓ PPO algorithm created successfully")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Clip param: {args.clip_param}")
        print(f"  - Gamma: {args.gamma}")
        print(f"  - Lambda: {args.lam}")
        
    except Exception as e:
        print(f"❌ PPO creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize storage
    print(f"\n💾 Initializing rollout storage...")
    try:
        # Get observation shapes for storage initialization
        dummy_obs = env.get_observations()
        actions_shape = (env.num_actions,)
        
        # Initialize PPO storage  
        ppo.init_storage(
            "rl",  # training_type for reinforcement learning
            env.num_envs,
            args.num_steps_per_env,
            dummy_obs,
            actions_shape
        )
        
        print(f"✓ Rollout storage initialized successfully")
        print(f"  - Num transitions per env: {args.num_steps_per_env}")
        print(f"  - Total storage size: {env.num_envs * args.num_steps_per_env}")
        
    except Exception as e:
        print(f"❌ Rollout storage initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Training loop
    print(f"\n🎯 Starting RSL-RL PPO training loop...")
    print(f"  - Total timesteps: {args.total_timesteps:,}")
    print(f"  - Max iterations: {args.max_iterations:,}")
    print(f"  - Steps per rollout: {args.num_steps_per_env}")
    print(f"  - Log interval: {args.log_interval}")
    print("-" * 80)
    
    total_steps = 0
    iteration = 0
    
    try:
        while total_steps < args.total_timesteps and iteration < args.max_iterations:
            # Get current observations
            obs = env.get_observations()
            
            # Collect rollout using RSL-RL PPO workflow
            for step in range(args.num_steps_per_env):
                # Get action from policy
                actions = ppo.act(obs)
                
                # Step environment
                obs, rewards, dones, extras = env.step(actions)
                
                # Process environment step for PPO
                ppo.process_env_step(obs, rewards, dones, extras)
                
                total_steps += env.num_envs
            
            # Compute returns and advantages
            ppo.compute_returns(obs)
            
            # Update policy
            loss_dict = ppo.update()
            
            iteration += 1
            
            # Logging
            if iteration % args.log_interval == 0:
                # Extract losses from loss_dict
                mean_value_loss = loss_dict.get('value_loss', 0.0)
                mean_surrogate_loss = loss_dict.get('surrogate_loss', 0.0)
                mean_entropy_loss = loss_dict.get('entropy_loss', 0.0)
                
                # Get reward statistics from storage if available
                if hasattr(ppo.storage, 'rewards'):
                    mean_reward = torch.mean(ppo.storage.rewards).item()
                else:
                    mean_reward = 0.0
                
                # Extract episode statistics from last extras
                goals_achieved = extras.get('goal_achieved', 0.0)
                episode_length = extras.get('episode_length', 0)
                distance_to_goal = extras.get('distance_to_goal', 0.0)
                
                # Calculate success rate and other metrics
                success_rate = float(goals_achieved) if episode_length > 0 else 0.0
                actual_episode_length = episode_length if episode_length > 0 else env.episode_length_buf[0].item()
                
                print(f"Iter {iteration:4d} | Steps {total_steps:8d} | "
                      f"Reward {mean_reward:6.3f} | Ep Len {actual_episode_length:6.1f} | "
                      f"Success {success_rate:.3f} | Value Loss {mean_value_loss:6.4f}")
                
                # Create log data for both wandb and CSV
                log_data = {
                    'iteration': iteration,
                    'total_steps': total_steps,
                    'rewards/mean': mean_reward,
                    'episode_length/actual': actual_episode_length,
                    'episode_length/buffer': env.episode_length_buf[0].item(),
                    'success_rate': success_rate,
                    'goals_achieved': goals_achieved,
                    'distance_to_goal': distance_to_goal,
                    'losses/value_loss': mean_value_loss,
                    'losses/surrogate_loss': mean_surrogate_loss,
                    'losses/entropy_loss': mean_entropy_loss,
                    'config/reward_type': args.reward_type,
                    'config/env_type': args.env_type,
                }
                
                if wandb_initialized:
                    wandb.log(log_data)
                
                # Always log to CSV for incremental backup
                _log_to_csv(log_data, args.experiment_name)
        
        print(f"\n✅ Training completed!")
        print(f"  - Total steps: {total_steps:,}")
        print(f"  - Total iterations: {iteration}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save model
    print(f"\n💾 Saving model...")
    try:
        models_dir = Path('models') / args.experiment_name
        models_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor_critic_state_dict': policy.state_dict(),
            'ppo_state_dict': ppo.optimizer.state_dict(),
            'args': args,
            'total_steps': total_steps,
            'iteration': iteration,
        }, models_dir / 'final_model.pt')
        
        print(f"✓ Model saved to {models_dir}")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
    
    if wandb_initialized:
        wandb.finish()
    
    print(f"\n🎉 Goal Navigation Experiment completed: {args.experiment_name}")


if __name__ == '__main__':
    main()