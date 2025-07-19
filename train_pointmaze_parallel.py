#!/usr/bin/env python3
"""
Parallelized PointMaze Training Script
=====================================

This script trains FastTD3 on multiple parallel PointMaze environments to fully utilize GPU memory.
Based on the fast_td3 reference implementation with proper vectorization.

Usage:
    python train_pointmaze_parallel.py --num-envs 1024 --total-timesteps 1000000
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Import ogbench to register environments
import ogbench
from ogbench.wrappers import RelativeGoalWrapper, SpeedWrapper

import sys
sys.path.append('fasttd3/fast_sac')

from fast_sac_utils import SimpleReplayBuffer, EmpiricalNormalization
from fast_sac import Actor, Critic
from tensordict import TensorDict

@dataclass
class ParallelTrainingArgs:
    """Arguments for parallel PointMaze training"""
    env_name: str = "pointmaze-medium-v0"  # Fixed: Use correct ogbench PointMaze name
    num_envs: int = 1024  # Start with 1024 parallel environments
    total_timesteps: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256  # Fixed: Batch size per environment (will be scaled by num_envs)
    buffer_size: int = 1_000_000
    
    # Learning hyperparameters (from fast_td3 reference)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.1
    policy_noise: float = 0.001
    noise_clip: float = 0.5
    policy_frequency: int = 2
    
    # Network architecture (from fast_td3 reference)
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 1024
    init_scale: float = 0.01
    
    # Training settings
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_interval: int = 25_000
    save_interval: int = 100_000
    log_interval: int = 1000
    
    # Fast SAC specific
    obs_normalization: bool = True
    num_updates: int = 2
    max_grad_norm: float = 0.0


class VectorizedPointMazeEnv:
    """Vectorized PointMaze environment wrapper for parallel training"""
    
    def __init__(self, env_name: str, num_envs: int, device: torch.device):
        self.env_name = env_name
        self.num_envs = num_envs
        self.device = device
        
        # Create a single environment to get dimensions
        import gymnasium as gym
        single_env = gym.make(env_name)
        single_env = SpeedWrapper(single_env, speed_multiplier=3.0)
        single_env = RelativeGoalWrapper(single_env)
        
        self.observation_space = single_env.observation_space
        self.action_space = single_env.action_space
        self.max_episode_steps = single_env._max_episode_steps if hasattr(single_env, '_max_episode_steps') else 500
        
        # Store dimensions
        self.num_obs = self.observation_space.shape[0]
        self.num_actions = self.action_space.shape[0]
        self.asymmetric_obs = False
        
        single_env.close()
        
        # Create parallel environments
        self.envs = []
        for i in range(num_envs):
            env = gym.make(env_name)
            env = SpeedWrapper(env, speed_multiplier=3.0)
            env = RelativeGoalWrapper(env)
            self.envs.append(env)
        
        # Initialize episode tracking
        self.episode_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
        print(f"✅ Created {num_envs} parallel PointMaze environments")
        print(f"   Observation space: {self.observation_space}")
        print(f"   Action space: {self.action_space}")
        
    def reset(self) -> torch.Tensor:
        """Reset all environments"""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        
        # Fix: Convert to numpy array first, then to tensor
        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        self.episode_steps.fill_(0)
        self.episode_returns.fill_(0.0)
        
        return observations
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step all environments in parallel"""
        actions_np = actions.cpu().numpy()
        
        next_observations = []
        rewards = []
        dones = []
        infos = {"time_outs": []}
        
        for i, (env, action) in enumerate(zip(self.envs, actions_np)):
            obs, reward, terminated, truncated, info = env.step(action)
            
            next_observations.append(obs)
            rewards.append(reward)
            
            # Handle episode termination
            done = terminated or truncated
            dones.append(done)
            infos["time_outs"].append(truncated)
            
            # Update episode tracking
            self.episode_steps[i] += 1
            self.episode_returns[i] += reward
            
            # Reset environment if episode is done
            if done:
                obs, _ = env.reset()
                next_observations[-1] = obs
                self.episode_steps[i] = 0
                self.episode_returns[i] = 0.0
        
        # Fix: Convert to numpy array first, then to tensor
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        infos["time_outs"] = torch.tensor(infos["time_outs"], dtype=torch.bool, device=self.device)
        
        return next_observations, rewards, dones, infos


def create_networks(args: ParallelTrainingArgs, obs_dim: int, act_dim: int) -> Tuple[Actor, Critic, Critic]:
    """Create actor and critic networks for SAC"""
    
    # Actor network (SAC doesn't use actor target)
    actor = Actor(
        obs_dim, act_dim, 
        num_envs=args.num_envs, 
        device=args.device, 
        init_scale=args.init_scale, 
        hidden_dim=args.actor_hidden_dim
    ).to(args.device)
    
    # Critic networks
    critic = Critic(
        obs_dim, act_dim, 
        hidden_dim=args.critic_hidden_dim, 
        device=args.device
    ).to(args.device)
    
    critic_target = Critic(
        obs_dim, act_dim, 
        hidden_dim=args.critic_hidden_dim, 
        device=args.device
    ).to(args.device)
    critic_target.load_state_dict(critic.state_dict())
    
    return actor, critic, critic_target


def update_networks(
    actor: Actor, 
    critic: Critic, 
    critic_target: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    target_entropy: float,
    batch: Dict[str, torch.Tensor],
    args: ParallelTrainingArgs,
    global_step: int
) -> Dict[str, float]:
    """Update actor and critic networks using SAC"""
    
    obs = batch["observations"]
    next_obs = batch["next"]["observations"]
    actions = batch["actions"]
    rewards = batch["next"]["rewards"]
    dones = batch["next"]["dones"]
    
    # Update critic
    with torch.no_grad():
        # SAC: Get next actions and log probs from actor
        next_action, next_log_prob, _ = actor(next_obs)
        target_q1, target_q2 = critic_target(next_obs, next_action)
        target_q = torch.min(target_q1, target_q2)
        # SAC: Subtract log prob for entropy regularization
        target_q = rewards + args.gamma * (1 - dones.float()) * (target_q - log_alpha.exp() * next_log_prob)
    
    # Current Q-values
    current_q1, current_q2 = critic(obs, actions)
    
    # Critic loss
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    
    # Update critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
    critic_optimizer.step()
    
    logs = {"critic_loss": critic_loss.item()}
    
    # Update actor (delayed)
    if global_step % args.policy_frequency == 0:
        # SAC: Actor update
        action, log_prob, _ = actor(obs)
        q1, q2 = critic(obs, action)
        q_value = torch.min(q1, q2)
        actor_loss = (log_alpha.exp() * log_prob - q_value).mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        actor_optimizer.step()
        
        # SAC: Alpha update
        alpha_loss = -log_alpha.exp() * (log_prob + target_entropy).detach().mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        
        # SAC: Only update critic target (no actor target in SAC)
        with torch.no_grad():
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        logs["actor_loss"] = actor_loss.item()
        logs["alpha_loss"] = alpha_loss.item()
        logs["alpha"] = log_alpha.exp().item()
    
    return logs


def evaluate_policy(
    actor: Actor, 
    obs_normalizer: EmpiricalNormalization, 
    args: ParallelTrainingArgs, 
    num_eval_episodes: int = 10
) -> Tuple[float, float]:
    """Evaluate the current policy"""
    import gymnasium as gym
    
    # Create evaluation environment
    env = gym.make(args.env_name)
    env = SpeedWrapper(env, speed_multiplier=3.0)
    env = RelativeGoalWrapper(env)
    
    episode_returns = []
    episode_lengths = []
    
    for episode in range(num_eval_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < env._max_episode_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            
            with torch.no_grad():
                norm_obs = obs_normalizer(obs_tensor).squeeze(0)
                action, _, _ = actor(norm_obs.unsqueeze(0))
                action = action.cpu().numpy()[0]
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    env.close()
    
    return np.mean(episode_returns), np.mean(episode_lengths)


def main():
    parser = argparse.ArgumentParser(description="Parallel PointMaze Training")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--batch-size", type=int, default=32768, help="Batch size for training")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Timesteps before learning starts")
    parser.add_argument("--eval-interval", type=int, default=25_000, help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=100_000, help="Model save interval")
    
    cmd_args = parser.parse_args()
    
    # Create training arguments
    args = ParallelTrainingArgs()
    args.num_envs = cmd_args.num_envs
    args.total_timesteps = cmd_args.total_timesteps
    args.batch_size = cmd_args.batch_size
    args.learning_starts = cmd_args.learning_starts
    args.eval_interval = cmd_args.eval_interval
    args.save_interval = cmd_args.save_interval
    
    print(f"🚀 Starting Parallel PointMaze Training")
    print(f"   Device: {args.device}")
    print(f"   Parallel Environments: {args.num_envs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Total Timesteps: {args.total_timesteps:,}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create vectorized environment
    device = torch.device(args.device)
    envs = VectorizedPointMazeEnv(args.env_name, args.num_envs, device)
    
    # Create networks
    actor, critic, critic_target = create_networks(
        args, envs.num_obs, envs.num_actions
    )
    
    # Create optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    
    # SAC: Alpha parameter for entropy regularization
    target_entropy = -float(envs.num_actions)
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(0.001)], device=device))
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.critic_learning_rate)
    
    # Create observation normalizer
    obs_normalizer = EmpiricalNormalization(shape=envs.num_obs, device=device)
    
    # Create replay buffer
    replay_buffer = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=envs.num_obs,
        n_act=envs.num_actions,
        n_critic_obs=envs.num_obs,
        asymmetric_obs=False,
        device=device,
    )
    
    # Initialize environment
    obs = envs.reset()
    global_step = 0
    episode_count = 0
    best_eval_return = -float('inf')
    
    print(f"\n🎯 Training Started!")
    print(f"   Observation shape: {obs.shape}")
    print(f"   GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    start_time = time.time()
    
    while global_step < args.total_timesteps:
        # Collect experience
        with torch.no_grad():
            # Normalize observations
            norm_obs = obs_normalizer(obs)
            
            # Get actions
            if global_step < args.learning_starts:
                # Random actions during initial exploration
                actions = torch.rand(args.num_envs, envs.num_actions, device=device) * 2 - 1
            else:
                # Policy actions from SAC actor
                actions, _, _ = actor(norm_obs)
        
        # Step environment
        next_obs, rewards, dones, infos = envs.step(actions)
        
        # Store transitions
        transition = TensorDict(
            {
                "observations": obs,
                "actions": actions,
                "next": {
                    "observations": next_obs,
                    "rewards": rewards,
                    "dones": dones.long(),
                    "truncations": infos["time_outs"].long(),
                },
            },
            batch_size=(args.num_envs,),
            device=device,
        )
        replay_buffer.extend(transition)
        
        # Update observation normalizer
        obs_normalizer.update(obs)
        
        obs = next_obs
        global_step += args.num_envs
        
        # Training updates
        if global_step >= args.learning_starts:
            for _ in range(args.num_updates):
                # Use a reasonable batch size that won't cause OOM
                # For small env counts, use the full batch_size, for large env counts, scale it down
                effective_batch_size = min(args.batch_size, max(64, args.batch_size // max(1, args.num_envs // 64)))
                batch = replay_buffer.sample(effective_batch_size)
                
                # Normalize batch observations
                batch["observations"] = obs_normalizer(batch["observations"])
                batch["next"]["observations"] = obs_normalizer(batch["next"]["observations"])
                
                logs = update_networks(
                    actor, critic, critic_target,
                    actor_optimizer, critic_optimizer, alpha_optimizer,
                    log_alpha, target_entropy,
                    batch, args, global_step
                )
        
        # Logging
        if global_step % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
            
            print(f"Step {global_step:,}/{args.total_timesteps:,} | "
                  f"SPS: {steps_per_sec:.0f} | "
                  f"GPU: {gpu_memory:.2f}GB | "
                  f"Mean Reward: {rewards.mean().item():.3f}")
        
        # Evaluation
        if global_step % args.eval_interval == 0 and global_step >= args.learning_starts:
            eval_return, eval_length = evaluate_policy(actor, obs_normalizer, args)
            print(f"\n📊 Evaluation at step {global_step:,}:")
            print(f"   Mean Return: {eval_return:.3f}")
            print(f"   Mean Length: {eval_length:.1f}")
            
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                print(f"   🎉 New best return: {best_eval_return:.3f}")
        
        # Save model
        if global_step % args.save_interval == 0 and global_step >= args.learning_starts:
            checkpoint = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "obs_normalizer_state": obs_normalizer.state_dict(),
                "global_step": global_step,
                "best_eval_return": best_eval_return,
                "args": args
            }
            torch.save(checkpoint, f"pointmaze_parallel_step_{global_step}.pt")
            print(f"💾 Model saved at step {global_step:,}")
    
    # Final evaluation
    final_eval_return, final_eval_length = evaluate_policy(actor, obs_normalizer, args, num_eval_episodes=20)
    
    total_time = time.time() - start_time
    print(f"\n🏁 Training Complete!")
    print(f"   Total Time: {total_time:.1f} seconds")
    print(f"   Final Return: {final_eval_return:.3f}")
    print(f"   Best Return: {best_eval_return:.3f}")
    print(f"   Final GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    # Save final model
    final_checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "obs_normalizer_state": obs_normalizer.state_dict(),
        "global_step": global_step,
        "best_eval_return": best_eval_return,
        "final_eval_return": final_eval_return,
        "args": args
    }
    torch.save(final_checkpoint, "pointmaze_parallel_final.pt")
    print("💾 Final model saved as pointmaze_parallel_final.pt")


if __name__ == "__main__":
    main() 