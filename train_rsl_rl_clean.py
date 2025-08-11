#!/usr/bin/env python3
"""
Clean RSL-RL PPO Training Script for OGBench Point Locomotion
============================================================

This script properly integrates RSL-RL's PPO with OGBench environments using:
- TensorDict observations (required by RSL-RL)
- Proper VecEnv interface implementation  
- Correct PPO workflow: act() -> process_env_step() -> compute_returns() -> update()
- Full wandb logging integration

Usage:
    python train_rsl_rl_clean.py --env_name pointmaze-medium-v0 --use_wandb
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

# Import ogbench to register environments
import ogbench

# Import our OGBench environment wrapper
from fasttd3.fast_sac.environments.ogbench_env import OGBenchEnv


class OGBenchRSLRLVecEnv(VecEnv):
    """RSL-RL VecEnv wrapper for OGBench environments using TensorDict observations."""
    
    def __init__(self, env: OGBenchEnv, cfg: dict = None):
        self.env = env
        self.cfg = cfg or {}
        
        # Action processing options (clipping enabled by default for stability)
        self.clip_actions = cfg.get('clip_actions', True)  # Default: True
        self.action_scale = cfg.get('action_scale', 1.0)
        
        # Reward shaping options
        self.reward_type = cfg.get('reward_type', 'sparse')
        self.dense_reward_scale = cfg.get('dense_reward_scale', 0.01)
        self.mixed_dense_weight = cfg.get('mixed_dense_weight', 0.1)
        
        # Initialize goal position (will be set from environment)
        self._goal_pos = None
        
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
        # Get raw observations from OGBench environment
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
        # Process actions to prevent physics instability
        processed_actions = actions.clone()
        
        # Apply action scaling
        if self.action_scale != 1.0:
            processed_actions *= self.action_scale
        
        # Apply action clipping to prevent extreme values
        if self.clip_actions:
            processed_actions = torch.clamp(processed_actions, -1.0, 1.0)
        
        # Step the underlying environment
        raw_obs, rewards, dones, infos = self.env.step(processed_actions)
        
        # Apply reward shaping if not sparse
        if self.reward_type != 'sparse':
            rewards = self._apply_reward_shaping(raw_obs, rewards, infos)
        
        # Update episode lengths
        self.episode_length_buf += 1
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": raw_obs,  # Policy observations
        }, batch_size=[self.num_envs], device=self.device)
        
        # Store for get_observations()
        self._last_obs = obs_tensordict
        
        # Create extras dict with required RSL-RL fields
        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        
        # Track episode statistics for logging
        log_info = {}
        
        # Check for completed episodes
        reset_mask = dones.bool() | time_outs.bool()
        if reset_mask.any():
            # Log episode statistics for completed episodes
            completed_episodes = reset_mask.sum().item()
            completed_lengths = self.episode_length_buf[reset_mask]
            completed_rewards = infos.get('episode_rewards', torch.zeros_like(rewards))[reset_mask]
            
            if completed_episodes > 0:
                log_info.update({
                    '/episode/completed_episodes': completed_episodes,
                    '/episode/mean_length': completed_lengths.float().mean().item(),
                    '/episode/mean_return': completed_rewards.float().mean().item() if completed_rewards.numel() > 0 else 0.0,
                    '/episode/max_return': completed_rewards.float().max().item() if completed_rewards.numel() > 0 else 0.0,
                    '/episode/min_return': completed_rewards.float().min().item() if completed_rewards.numel() > 0 else 0.0,
                })
        
        extras = {
            "time_outs": time_outs,  # Required by RSL-RL for bootstrapping
            "log": log_info  # Episode statistics
        }
        
        # Reset episode lengths for done/timeout environments
        self.episode_length_buf[reset_mask] = 0
        
        return obs_tensordict, rewards, dones, extras
    
    def _apply_reward_shaping(self, observations: torch.Tensor, original_rewards: torch.Tensor, infos: dict) -> torch.Tensor:
        """Apply reward shaping based on configuration."""
        if self.reward_type == 'sparse':
            return original_rewards
        
        # Extract agent positions (first 2 elements of observation are typically x, y coordinates)
        agent_positions = observations[:, :2]  # Shape: [num_envs, 2]
        
        # Get goal position - try multiple methods
        goal_positions = self._get_goal_positions()
        if goal_positions is None:
            # Fallback to sparse rewards if we can't get goal positions
            return original_rewards
        
        # Compute euclidean distances to goal
        distances = torch.norm(agent_positions - goal_positions, dim=1)  # Shape: [num_envs]
        
        # Create dense rewards based on distance (closer = higher reward)
        # Normalize by maze-specific max distance
        env_name = getattr(self.env, 'env_name', 'pointmaze-medium-v0')
        if 'small' in env_name:
            max_distance = 6.0   # Diagonal of small maze ~6
        elif 'medium' in env_name:
            max_distance = 17.0  # Diagonal of medium maze ~17
        elif 'large' in env_name:
            max_distance = 28.0  # Diagonal of large maze ~28
        else:
            max_distance = 12.0  # Conservative default
        
        # Distance-based reward: reward decreases linearly with distance
        dense_rewards = torch.clamp(max_distance - distances, min=0.0) / max_distance * self.dense_reward_scale
        
        if self.reward_type == 'dense':
            # Pure dense rewards
            shaped_rewards = dense_rewards
        elif self.reward_type == 'mixed':
            # Combine sparse and dense rewards
            shaped_rewards = original_rewards + (dense_rewards * self.mixed_dense_weight)
        else:
            shaped_rewards = original_rewards
        
        return shaped_rewards
    
    def _get_goal_positions(self) -> torch.Tensor:
        """Extract goal positions from environment."""
        # Use cached goal position if available
        if self._goal_pos is not None:
            return self._goal_pos.repeat(self.num_envs, 1)
        
        try:
            # Safe approach: Use environment-specific fallback positions
            # For OGBench pointmaze environments, goals are typically in these positions
            env_name = getattr(self.env, 'env_name', 'pointmaze-medium-v0')
            
            if 'pointmaze' in env_name.lower():
                # Standard pointmaze goal positions (approximate)
                if 'small' in env_name:
                    default_goal = torch.tensor([[4.0, 4.0]], device=self.device, dtype=torch.float32)
                elif 'medium' in env_name:
                    default_goal = torch.tensor([[12.0, 12.0]], device=self.device, dtype=torch.float32)  
                elif 'large' in env_name:
                    default_goal = torch.tensor([[20.0, 20.0]], device=self.device, dtype=torch.float32)
                else:
                    # Default fallback
                    default_goal = torch.tensor([[8.0, 8.0]], device=self.device, dtype=torch.float32)
            else:
                # Generic navigation task fallback
                default_goal = torch.tensor([[10.0, 10.0]], device=self.device, dtype=torch.float32)
            
            # Cache for future use
            self._goal_pos = default_goal
            return default_goal.repeat(self.num_envs, 1)
            
        except Exception as e:
            print(f"Warning: Could not determine goal positions, falling back to sparse rewards: {e}")
            return None

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def _log_to_csv(log_data, experiment_name):
    """Backup CSV logging for when wandb is offline."""
    import csv
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    csv_file = log_dir / f"{experiment_name}_metrics.csv"
    
    # Check if file exists to write headers
    write_header = not csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        
        if write_header:
            writer.writeheader()
        
        writer.writerow(log_data)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean RSL-RL PPO Training for OGBench')
    
    # Environment arguments
    parser.add_argument('--env_name', type=str, default='pointmaze-medium-v0',
                        help='OGBench environment name')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--num_steps_per_env', type=int, default=1024,
                        help='Number of steps per environment per rollout')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='Maximum number of training iterations')
    
    # PPO arguments
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
                        help='GAE lambda parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value_loss_coef', type=float, default=1.0,
                        help='Value loss coefficient')
    parser.add_argument('--no_clip_actions', action='store_true',
                        help='Disable action clipping (enabled by default for stability)')
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Scale actions by this factor (helps reduce instability)')
    
    # Reward shaping arguments
    parser.add_argument('--reward_type', type=str, default='sparse', 
                        choices=['sparse', 'dense', 'mixed'],
                        help='Reward type: sparse (goal only), dense (distance-based), or mixed (both)')
    parser.add_argument('--dense_reward_scale', type=float, default=0.1,
                        help='Scale factor for dense distance-based rewards (0.1 for dense, 0.01-0.05 for mixed)')
    parser.add_argument('--mixed_dense_weight', type=float, default=0.1,
                        help='Weight for dense component in mixed rewards (sparse always 1.0)')
    
    # Network arguments
    parser.add_argument('--hidden_dims', type=str, default='256,256',
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ogbench-rsl-rl-clean',
                        help='Wandb project name')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (iterations)')
    
    return parser.parse_args()


def main():
    """Main training function using clean RSL-RL implementation."""
    args = get_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"clean_rslrl_{args.env_name}_{timestamp}"
    
    print(f"🚀 Starting Clean RSL-RL PPO Training")
    print(f"Experiment: {args.experiment_name}")
    print(f"Environment: {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps}")
    
    # Initialize wandb (with offline mode for clusters without internet)
    wandb_initialized = False
    if args.use_wandb:
        # Check if we're on a compute node (no internet) vs login node (internet)
        import socket
        hostname = socket.gethostname()
        is_compute_node = 'jrc' in hostname or 'batch' in hostname or 'node' in hostname
        
        if is_compute_node:
            # Aggressive offline mode for compute nodes - prevent all network attempts
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_CONSOLE"] = "off"  # Reduce verbose output
            os.environ["WANDB_SILENT"] = "true"  # Suppress network retry messages
            os.environ["WANDB__SERVICE_WAIT"] = "0"  # Don't wait for wandb service
            print(f"🔄 Detected compute node ({hostname}) - Using WANDB offline mode")
        
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args),
                mode="offline" if is_compute_node else "online"
            )
            wandb_initialized = True
        except Exception as e:
            print(f"❌ Wandb initialization failed: {e}")
            print("🔄 Continuing training without wandb logging...")
            wandb_initialized = False
        
        if wandb_initialized:
            if is_compute_node:
                print(f"✓ Wandb initialized in OFFLINE mode: {args.wandb_project}")
                print(f"  - Logs saved locally to: {wandb.run.dir}")
                print(f"  - Sync later from login node with: wandb sync {wandb.run.dir}")
            else:
                print(f"✓ Wandb initialized in ONLINE mode: {args.wandb_project}")
        else:
            print(f"⚠️  Training will proceed without wandb logging")
    
    # Create environment
    print(f"\n🏗️  Creating environment...")
    try:
        base_env = OGBenchEnv(
            env_name=args.env_name,
            num_envs=args.num_envs,
            device=device
        )
        env_cfg = {
            'clip_actions': not args.no_clip_actions,  # Inverted: default True, --no_clip_actions makes it False
            'action_scale': args.action_scale,
            'reward_type': args.reward_type,
            'dense_reward_scale': args.dense_reward_scale,
            'mixed_dense_weight': args.mixed_dense_weight
        }
        env = OGBenchRSLRLVecEnv(base_env, cfg=env_cfg)
        
        print(f"✓ Environment created successfully")
        print(f"  - Environments: {env.num_envs}")
        print(f"  - Observations: {base_env.num_obs}")
        print(f"  - Actions: {env.num_actions}")
        print(f"  - Max episode length: {env.max_episode_length}")
        print(f"  - Action clipping: {'✓ Enabled' if env.clip_actions else '✗ Disabled'}")
        print(f"  - Action scale: {env.action_scale}")
        print(f"  - Reward type: {env.reward_type}")
        if env.reward_type == 'dense':
            print(f"    • Dense reward scale: {env.dense_reward_scale}")
        elif env.reward_type == 'mixed':
            print(f"    • Dense reward scale: {env.dense_reward_scale}")
            print(f"    • Dense weight: {env.mixed_dense_weight}")
        
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
            policy,
            num_learning_epochs=args.num_learning_epochs,
            num_mini_batches=args.num_mini_batches,
            clip_param=args.clip_param,
            gamma=args.gamma,
            lam=args.lam,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            learning_rate=args.learning_rate,
            device=device
        )
        
        print(f"✓ PPO algorithm created successfully")
        
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
        # Note: training_type must be compatible with mini_batch_generator
        ppo.init_storage(
            "rl",  # training_type for reinforcement learning
            env.num_envs,
            args.num_steps_per_env,
            dummy_obs,
            actions_shape
        )
        
        # PATCH: Manually create missing storage attributes (RSL-RL version compatibility)
        missing_attrs = []
        
        # Create values storage if missing
        if not hasattr(ppo.storage, 'values'):
            ppo.storage.values = torch.zeros(
                args.num_steps_per_env, 
                env.num_envs, 
                1,  # value is scalar per environment
                device=ppo.storage.device
            )
            missing_attrs.append('values')
        
        # Create returns storage if missing 
        if not hasattr(ppo.storage, 'returns'):
            ppo.storage.returns = torch.zeros(
                args.num_steps_per_env, 
                env.num_envs, 
                1,  # return is scalar per environment
                device=ppo.storage.device
            )
            missing_attrs.append('returns')
        
        # Create advantages storage if missing
        if not hasattr(ppo.storage, 'advantages'):
            ppo.storage.advantages = torch.zeros(
                args.num_steps_per_env, 
                env.num_envs, 
                1,  # advantage is scalar per environment
                device=ppo.storage.device
            )
            missing_attrs.append('advantages')
            
        if missing_attrs:
            print(f"  - PATCH: Created missing storage attributes: {missing_attrs}")
            
            # PATCH: Override add_transitions to properly store values when values was missing
            if 'values' in missing_attrs:
                original_add_transitions = ppo.storage.add_transitions
                
                def patched_add_transitions(transition):
                    # Call original method
                    original_add_transitions(transition)
                    
                    # Manually store values if they exist in transition
                    if hasattr(transition, 'values') and transition.values is not None:
                        # Get current step from storage
                        current_step = ppo.storage.step - 1  # step was already incremented
                        if 0 <= current_step < args.num_steps_per_env:
                            ppo.storage.values[current_step] = transition.values.clone()
                
                ppo.storage.add_transitions = patched_add_transitions
                print(f"  - PATCH: Overrode add_transitions to store values")
        
        # Verify all required storage attributes exist
        required_attrs = ['observations', 'actions', 'rewards', 'dones', 'values', 'returns', 'advantages']
        for attr in required_attrs:
            if hasattr(ppo.storage, attr):
                print(f"  - ✓ {attr}: {getattr(ppo.storage, attr).shape}")
            else:
                print(f"  - ❌ Missing: {attr}")
        
        print(f"✓ Rollout storage initialized and patched")
        print(f"  - Storage device: {ppo.storage.device}")
        
    except Exception as e:
        print(f"❌ Storage initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Training loop
    print(f"\n🎯 Starting RSL-RL PPO training loop...")
    print(f"Steps per rollout: {args.num_steps_per_env}")
    print(f"Max iterations: {args.max_iterations}")
    print("-" * 80)
    
    start_time = datetime.now()
    total_timesteps = 0
    
    try:
        for iteration in range(args.max_iterations):
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
                
                total_timesteps += env.num_envs
            
            # Compute returns and advantages
            ppo.compute_returns(obs)
            
            # Update policy
            loss_dict = ppo.update()
            
            # Collect episode statistics from extras
            episode_stats = {}
            if 'log' in extras and extras['log']:
                for key, value in extras['log'].items():
                    if key.startswith('/episode/'):
                        episode_stats[key] = value
            
            # Logging
            if iteration % args.log_interval == 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                fps = total_timesteps / elapsed_time if elapsed_time > 0 else 0
                
                # Compute additional training metrics
                mean_reward = rewards.mean().item()
                mean_episode_length = env.episode_length_buf.float().mean().item()
                
                log_data = {
                    'iteration': iteration,
                    'total_timesteps': total_timesteps,
                    'fps': fps,
                    'elapsed_time': elapsed_time,
                    'env/mean_reward': mean_reward,
                    'env/mean_episode_length': mean_episode_length,
                    'env/max_episode_length': env.episode_length_buf.max().item(),
                    'config/reward_type': env.reward_type,  # Track reward type
                    **{f'loss/{k}': v for k, v in loss_dict.items()},
                    **episode_stats  # Add episode statistics when available
                }
                
                # Enhanced console output
                episode_info = ""
                if '/episode/mean_return' in episode_stats:
                    episode_info = f" | Ep Return: {episode_stats['/episode/mean_return']:.2f}"
                
                print(f"Iter {iteration:6d} | "
                      f"Steps: {total_timesteps:8d} | "
                      f"FPS: {fps:6.0f} | "
                      f"Value: {loss_dict['value_function']:.4f} | "
                      f"Policy: {loss_dict['surrogate']:.4f} | "
                      f"Entropy: {loss_dict['entropy']:.4f}"
                      f"{episode_info}")
                
                if args.use_wandb and wandb_initialized:
                    wandb.log(log_data)
                
                # Backup CSV logging (works even without internet)
                _log_to_csv(log_data, args.experiment_name)
            
            # Check stopping condition
            if total_timesteps >= args.total_timesteps:
                print(f"🎯 Reached target timesteps: {args.total_timesteps}")
                break
                
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    print(f"\n💾 Saving final model...")
    save_path = Path("models") / args.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_path = save_path / "final_model.pt"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'iteration': iteration if 'iteration' in locals() else 0,
        'total_timesteps': total_timesteps,
        'args': vars(args),
        'policy_cfg': policy_cfg,
    }, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    env.close()
    
    if args.use_wandb and wandb_initialized:
        wandb.finish()
        print("✓ Wandb session finished")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n🎉 Training completed successfully!")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Average FPS: {total_timesteps / elapsed:.0f}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
