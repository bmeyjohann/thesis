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
from environments.ogbench_env import OGBenchEnv
from simple_dynamic_pointmaze import SimpleDynamicPointMaze

# Register the simple environment
gym.register(
    id='SimpleDynamicPointMaze-v0',
    entry_point='simple_dynamic_pointmaze:SimpleDynamicPointMaze',
    max_episode_steps=500,
)


class SimpleDynamicRSLRLVecEnv(VecEnv):
    """RSL-RL VecEnv wrapper for SimpleDynamicPointMaze using TensorDict observations."""
    
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        
        # Environment configuration
        arena_size = self.cfg.get('arena_size', 20.0)
        max_episode_steps = self.cfg.get('max_episode_steps', 500)
        goal_threshold = self.cfg.get('goal_threshold', 0.5)
        action_scale = self.cfg.get('action_scale', 0.5)
        
        # Reward configuration  
        self.reward_type = self.cfg.get('reward_type', 'sparse')
        if self.reward_type == 'sparse':
            self.base_env = SimpleDynamicPointMaze(
                arena_size=arena_size,
                max_episode_steps=max_episode_steps,
                goal_reward=1.0,
                distance_reward_scale=0.0,  # No distance reward for sparse
                action_scale=action_scale,
                goal_threshold=goal_threshold,
            )
        else:  # dense
            self.base_env = SimpleDynamicPointMaze(
                arena_size=arena_size,
                max_episode_steps=max_episode_steps,
                goal_reward=1.0,
                distance_reward_scale=0.01,  # Distance reward for dense
                action_scale=action_scale,
                goal_threshold=goal_threshold,
            )
        
        # RSL-RL VecEnv required attributes
        self.num_envs = 1  # Single env for simple case
        self.num_actions = self.base_env.action_space.shape[0]
        self.max_episode_length = self.base_env.max_episode_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state
        self._last_obs = None
        
        # Initialize observations
        self.reset()
        
    def _enhance_observation(self, raw_obs, info):
        """Enhance simple environment observation with goal information."""
        # raw_obs: [agent_x, agent_y]
        # Add: [goal_x, goal_y, distance, direction_x, direction_y]
        
        agent_pos = raw_obs[:2]
        goal_pos = info['goal']
        
        # Calculate distance
        distance = np.linalg.norm(goal_pos - agent_pos)
        
        # Calculate normalized direction
        direction = goal_pos - agent_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        else:
            direction = np.zeros(2)
            
        # Combine all components
        enhanced_obs = np.concatenate([
            agent_pos,      # [0:2] Agent position
            goal_pos,       # [2:4] Goal position  
            [distance],     # [4] Distance to goal
            direction       # [5:7] Normalized direction
        ])
        
        return enhanced_obs.astype(np.float32)
        
    def get_observations(self) -> TensorDict:
        """Return current observations as TensorDict."""
        if self._last_obs is None:
            self.reset()
        return self._last_obs
    
    def reset(self) -> TensorDict:
        """Reset environment and return TensorDict observations."""
        # Reset the simple environment
        raw_obs, info = self.base_env.reset()
        
        # Enhance observation
        enhanced_obs = self._enhance_observation(raw_obs, info)
        obs_tensor = torch.from_numpy(enhanced_obs).unsqueeze(0).to(device=self.device, dtype=torch.float)
        
        # Convert to TensorDict with proper structure for RSL-RL
        self._last_obs = TensorDict({
            "policy": obs_tensor,  # Observations for policy network
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step environment with RSL-RL interface."""
        # Step the simple environment
        raw_obs, reward, terminated, truncated, info = self.base_env.step(actions[0].cpu().numpy())
        
        # Enhance observation
        enhanced_obs = self._enhance_observation(raw_obs, info)
        obs_tensor = torch.from_numpy(enhanced_obs).unsqueeze(0).to(device=self.device, dtype=torch.float)
        
        # Convert to tensors
        rewards = torch.tensor([reward], device=self.device, dtype=torch.float)
        dones = torch.tensor([terminated], device=self.device, dtype=torch.bool)
        
        # Update episode tracking
        self.episode_length_buf += 1
        
        # Create infos dict
        time_outs = torch.tensor([truncated], device=self.device, dtype=torch.bool)
        infos = {
            'time_outs': time_outs,
            'episode_rewards': rewards.clone(),  # For logging
        }
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[self.num_envs], device=self.device)
        
        self._last_obs = obs_tensordict
        
        return obs_tensordict, rewards, dones, infos


class OGBenchGoalRSLRLVecEnv(VecEnv):
    """RSL-RL VecEnv wrapper for OGBench environments with goal-enhanced observations."""
    
    def __init__(self, env: OGBenchEnv, cfg: dict = None):
        self.env = env
        self.cfg = cfg or {}
        
        # Reward shaping options
        self.reward_type = cfg.get('reward_type', 'sparse')
        self.dense_reward_scale = cfg.get('dense_reward_scale', 0.01)
        
        # RSL-RL VecEnv required attributes
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_steps
        self.device = env.sim_device
        
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
        # Get enhanced observations from OGBench environment
        raw_obs = self.env.reset()  # Shape: [num_envs, obs_dim]
        
        # Convert to TensorDict with proper structure for RSL-RL
        self._last_obs = TensorDict({
            "policy": raw_obs,  # Observations for policy network
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step environment with RSL-RL interface."""
        # Step the underlying environment
        raw_obs, rewards, dones, infos = self.env.step(actions)
        
        # Apply reward shaping if dense
        if self.reward_type == 'dense':
            # The OGBench wrapper already provides enhanced observations
            # We can add distance-based rewards here if needed
            pass  # For now, use sparse rewards from OGBench
        
        # Update episode tracking
        self.episode_length_buf += 1
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": raw_obs,  # Policy observations
        }, batch_size=[self.num_envs], device=self.device)
        
        # Store for get_observations()
        self._last_obs = obs_tensordict
        
        # Create extras dict with required RSL-RL fields
        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        
        # Update infos for RSL-RL compatibility
        rsl_infos = {
            'time_outs': time_outs,
            'episode_rewards': infos.get('episode_rewards', rewards.clone()),
        }
        
        return obs_tensordict, rewards, dones, rsl_infos


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
            env = OGBenchGoalRSLRLVecEnv(base_env, cfg=env_cfg)
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
                
                # Calculate episode statistics
                mean_episode_length = torch.mean(env.episode_length_buf.float()).item()
                
                # Get reward statistics from storage if available
                if hasattr(ppo.storage, 'rewards'):
                    mean_reward = torch.mean(ppo.storage.rewards).item()
                else:
                    mean_reward = 0.0
                
                print(f"Iter {iteration:4d} | Steps {total_steps:8d} | "
                      f"Reward {mean_reward:6.3f} | Ep Len {mean_episode_length:6.1f} | "
                      f"Value Loss {mean_value_loss:6.4f}")
                
                if wandb_initialized:
                    wandb.log({
                        'iteration': iteration,
                        'total_steps': total_steps,
                        'rewards/mean': mean_reward,
                        'episode_length/mean': mean_episode_length,
                        'losses/value_loss': mean_value_loss,
                        'losses/surrogate_loss': mean_surrogate_loss,
                        'losses/entropy_loss': mean_entropy_loss,
                    })
        
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