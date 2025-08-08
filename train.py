#!/usr/bin/env python3
"""
RSL-RL PPO Training Script for OGBench Point Locomotion
=====================================================

This script trains an agent using RSL-RL's PPO algorithm on OGBench point maze environments.
The environment wrapper is designed to be reusable with FastTD3 and other algorithms later.

Usage:
    python train.py --env_name pointmaze-medium-v0 --num_envs 1024 --total_timesteps 1000000
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import wandb
from pathlib import Path

# Add the fasttd3 directory to path for imports
sys.path.append('fasttd3/fast_sac')

# Import RSL-RL components
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.env import VecEnv

# Import our OGBench environment wrapper
from fasttd3.fast_sac.environments.ogbench_env import OGBenchEnv

# Import ogbench to register environments
import ogbench


class OGBenchVecEnvWrapper(VecEnv):
    """Wrapper to make OGBenchEnv compatible with RSL-RL's VecEnv interface."""
    
    def __init__(self, env: OGBenchEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_privileged_obs = None  # OGBench environments don't have privileged observations
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_steps
        self.device = env.sim_device
        
    def get_observations(self):
        """Return the current observations."""
        return self._last_obs
    
    def reset(self):
        """Reset the environment."""
        self._last_obs = self.env.reset()
        return self._last_obs
    
    def step(self, actions):
        """Step the environment."""
        obs, rewards, dones, infos = self.env.step(actions)
        self._last_obs = obs
        
        # RSL-RL expects rewards, dones to have proper shapes
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
            
        # Create extras dict for RSL-RL
        extras = {
            "time_outs": infos.get("time_outs", torch.zeros_like(dones)),
        }
        
        return obs, rewards, dones, extras
    
    def close(self):
        """Close the environment."""
        self.env.close()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RSL-RL PPO Training for OGBench')
    
    # Environment arguments
    parser.add_argument('--env_name', type=str, default='pointmaze-medium-v0',
                        help='OGBench environment name')
    parser.add_argument('--num_envs', type=int, default=256,
                        help='Number of parallel environments')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for policy and value networks')
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
    
    # Network arguments
    parser.add_argument('--hidden_dims', type=str, default='256,256',
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (in policy updates)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Model saving interval (in policy updates)')
    parser.add_argument('--eval_interval', type=int, default=250,
                        help='Evaluation interval (in policy updates)')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='Maximum number of policy updates')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ogbench-rsl-rl',
                        help='Wandb project name')
    
    return parser.parse_args()


def create_policy_config(args, env):
    """Create policy configuration for RSL-RL."""
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    policy_cfg = {
        'init_noise_std': 1.0,
        'actor_hidden_dims': hidden_dims,
        'critic_hidden_dims': hidden_dims,
        'activation': args.activation,
    }
    
    return policy_cfg


def create_algorithm_config(args):
    """Create algorithm configuration for RSL-RL PPO."""
    alg_cfg = {
        'class_name': 'PPO',
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
    
    return alg_cfg


def evaluate_policy(env, policy, num_eval_episodes=10):
    """Evaluate the policy."""
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                actions = policy.act_inference(obs)
            obs, rewards, dones, _ = env.step(actions)
            
            episode_reward += rewards.sum().item()
            episode_length += 1
            done = dones.any().item()
        
        episode_rewards.append(episode_reward / env.num_envs)
        episode_lengths.append(episode_length)
    
    policy.train()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }


def main():
    """Main training function."""
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
        args.experiment_name = f"ogbench_{args.env_name}_ppo_{timestamp}"
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Create environment
    print(f"Creating environment: {args.env_name}")
    base_env = OGBenchEnv(
        env_name=args.env_name,
        num_envs=args.num_envs,
        device=device
    )
    env = OGBenchVecEnvWrapper(base_env)
    
    print(f"Environment created with {env.num_envs} parallel environments")
    print(f"Observation space: {env.num_obs}")
    print(f"Action space: {env.num_actions}")
    
    # Create policy
    policy_cfg = create_policy_config(args, env)
    policy = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        **policy_cfg
    ).to(device)
    
    # Create algorithm
    alg_cfg = create_algorithm_config(args)
    ppo = PPO(policy, device=device, **alg_cfg)
    
    # Initialize policy
    obs = env.reset()
    
    # Training loop
    print("Starting training...")
    start_time = datetime.now()
    
    for iteration in range(args.max_iterations):
        # Collect rollouts
        obs = env.get_observations()
        
        # Step the environment for one rollout
        for step in range(ppo.data_loader.batch_size):
            actions = policy.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            ppo.data_loader.add_transitions(obs, actions, rewards, dones, extras)
        
        # Compute returns and advantages
        last_values = policy.evaluate(obs)
        ppo.data_loader.compute_returns(last_values, gamma=args.gamma, lam=args.lam)
        
        # Update policy
        mean_value_loss, mean_surrogate_loss, mean_entropy_loss = ppo.update()
        
        # Logging
        if iteration % args.log_interval == 0:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            timesteps = iteration * args.num_envs * ppo.data_loader.batch_size
            
            log_data = {
                'iteration': iteration,
                'timesteps': timesteps,
                'value_loss': mean_value_loss,
                'surrogate_loss': mean_surrogate_loss,
                'entropy_loss': mean_entropy_loss,
                'elapsed_time': elapsed_time,
                'fps': timesteps / elapsed_time if elapsed_time > 0 else 0,
            }
            
            print(f"Iteration {iteration:6d} | "
                  f"Timesteps: {timesteps:8d} | "
                  f"Value Loss: {mean_value_loss:.4f} | "
                  f"Policy Loss: {mean_surrogate_loss:.4f} | "
                  f"Entropy: {mean_entropy_loss:.4f} | "
                  f"FPS: {log_data['fps']:.0f}")
            
            if args.use_wandb:
                wandb.log(log_data)
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            eval_results = evaluate_policy(env, policy, num_eval_episodes=5)
            print(f"Evaluation | Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f} | "
                  f"Mean Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            
            if args.use_wandb:
                wandb.log({
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/std_reward': eval_results['std_reward'],
                    'eval/mean_length': eval_results['mean_length'],
                    'eval/std_length': eval_results['std_length'],
                })
        
        # Save model
        if iteration % args.save_interval == 0 and iteration > 0:
            save_path = Path(f"models/{args.experiment_name}")
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
                'args': vars(args),
            }, save_path / f"model_{iteration}.pt")
            print(f"Model saved to {save_path / f'model_{iteration}.pt'}")
        
        # Check if we've reached the target timesteps
        if iteration * args.num_envs * ppo.data_loader.batch_size >= args.total_timesteps:
            print(f"Reached target timesteps: {args.total_timesteps}")
            break
    
    # Final save
    save_path = Path(f"models/{args.experiment_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'iteration': iteration,
        'args': vars(args),
    }, save_path / "final_model.pt")
    print(f"Final model saved to {save_path / 'final_model.pt'}")
    
    # Close environment
    env.close()
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
