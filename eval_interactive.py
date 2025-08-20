#!/usr/bin/env python3
"""
Interactive Evaluation Script for RSL-RL Trained Agents

Load a trained RSL-RL policy and run it interactively on OGBench environments
with visual rendering. Perfect for testing trained agents and debugging!

Usage:
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-medium-v0
    
Controls:
    - ESC: Exit
    - SPACE: Reset environment
    - R: Toggle manual reset mode
    - Q: Quit
"""

import os
import sys
import argparse
import torch
import numpy as np
import gymnasium as gym
import pygame
from pathlib import Path
from datetime import datetime

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Import ogbench to register environments
import ogbench

# Custom environments will be registered through OGBench

# Add RSL-RL to path
sys.path.append('fasttd3/fast_sac')

# Import RSL-RL components
try:
    from rsl_rl.modules import ActorCritic  # ActorCritic is in modules, not algorithms!
    from tensordict import TensorDict
    print("✓ RSL-RL imports successful")
except ImportError as e:
    print(f"❌ RSL-RL import failed: {e}")
    print("Make sure RSL-RL is installed: pip install rsl_rl")
    sys.exit(1)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive evaluation of trained RSL-RL agents')
    
    # Model and environment
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pt file)')
    parser.add_argument('--env_name', type=str, default='pointmaze-medium-v0',
                        help='OGBench environment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    
    # Visualization
    parser.add_argument('--render_mode', type=str, default='human',
                        choices=['human', 'rgb_array'],
                        help='Rendering mode')
    parser.add_argument('--width', type=int, default=800,
                        help='Render width')
    parser.add_argument('--height', type=int, default=600,
                        help='Render height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS for rendering')
    
    # Evaluation
    parser.add_argument('--max_episode_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run (0 = infinite)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Action processing (should match training settings)
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Action scaling factor (should match training)')
    parser.add_argument('--clip_actions', action='store_true', default=True,
                        help='Clip actions to [-1, 1] (should match training)')
    
    return parser.parse_args()

def load_trained_policy(model_path: str, env, device: torch.device):
    """Load a trained RSL-RL policy from checkpoint."""
    print(f"🔄 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint (trust local files)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print(f"✓ Checkpoint loaded")
    
    # Create dummy observation to match model architecture
    dummy_obs = torch.zeros(1, env.observation_space.shape[0], device=device)
    dummy_obs_dict = TensorDict({
        "policy": dummy_obs,
    }, batch_size=[1], device=device)
    
    # Extract model configuration (if available)
    if 'policy_cfg' in checkpoint:
        config = checkpoint['policy_cfg']
        print(f"📋 Model config found: {config}")
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"📋 Model config found: {config}")
    else:
        # Default configuration (adjust based on your training settings)
        config = {
            'hidden_dims': [256, 256, 256],
            'activation': 'elu',
        }
        print(f"⚠️  No model config found, using defaults: {config}")
        
    # Clean config - remove keys that ActorCritic doesn't recognize
    valid_config_keys = {'hidden_dims', 'activation', 'init_noise_std', 'actor_hidden_dims', 'critic_hidden_dims'}
    config = {k: v for k, v in config.items() if k in valid_config_keys}
    
    # Create ActorCritic with proper observation groups
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"]
    }
    
    try:
        policy = ActorCritic(
            obs=dummy_obs_dict,
            obs_groups=obs_groups,
            num_actions=env.action_space.shape[0],
            **config
        ).to(device)
        print(f"✓ ActorCritic created with obs shape: {dummy_obs.shape}, action dim: {env.action_space.shape[0]}")
    except Exception as e:
        print(f"❌ ActorCritic creation failed: {e}")
        raise
    
    # Load model weights - handle different checkpoint formats
    if 'policy_state_dict' in checkpoint:
        # RSL-RL training checkpoint format (from your training script)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"✓ Model weights loaded (policy_state_dict)")
    elif 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model weights loaded (model_state_dict)")
    elif 'state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['state_dict'])
        print(f"✓ Model weights loaded (state_dict)")
    else:
        # Assume the checkpoint is the state dict itself
        try:
            policy.load_state_dict(checkpoint)
            print(f"✓ Model weights loaded (direct state dict)")
        except Exception as e:
            print(f"❌ Failed to load weights. Checkpoint keys: {list(checkpoint.keys())}")
            raise e
    
    policy.eval()  # Set to evaluation mode
    print(f"✓ Policy set to evaluation mode")
    
    # Print training info if available
    training_info = {}
    
    # Extract training info from different possible keys
    if 'training_info' in checkpoint:
        training_info.update(checkpoint['training_info'])
    
    # Additional info from checkpoint metadata
    for key in ['iteration', 'total_timesteps']:
        if key in checkpoint:
            training_info[key] = checkpoint[key]
    
    # Extract args if available
    if 'args' in checkpoint:
        args = checkpoint['args']
        if hasattr(args, 'env_name'):
            training_info['env_name'] = args.env_name
        if hasattr(args, 'reward_type'):
            training_info['reward_type'] = args.reward_type
    
    if training_info:
        print(f"📊 Training info:")
        for key, value in training_info.items():
            print(f"   {key}: {value}")
    else:
        print(f"ℹ️  No training info available in checkpoint")
    
    return policy

def create_env(env_name: str, args):
    """Create the evaluation environment with appropriate wrappers."""
    print(f"🏗️  Creating environment: {env_name}")
    
    # Base environment creation parameters
    env_kwargs = {
        'render_mode': args.render_mode,
        'max_episode_steps': args.max_episode_steps,
    }
    
    # Add width/height parameters for OGBench environments that support them
    if env_name.startswith(('pointmaze-', 'antmaze-', 'humanoidmaze-')):
        env_kwargs.update({
            'width': args.width,
            'height': args.height
        })
    
    try:
        env = gym.make(env_name, **env_kwargs)
        
        # Apply flexible observation wrapper for arena environments
        if env_name == 'pointmaze-arena-v0':
            from ogbench.wrappers import FlexibleObsWrapper
            env = FlexibleObsWrapper(env, include_goal=True)  # Match training config
            print(f"✓ Applied FlexibleObsWrapper")
        
        print(f"✓ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        return env
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        raise

def run_interactive_evaluation(policy, env, args, device):
    """Run interactive evaluation loop."""
    print(f"\n🎮 Starting Interactive Evaluation")
    print(f"Environment: {args.env_name}")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"\n🎮 Controls:")
    print(f"   ESC/Q: Exit")
    print(f"   SPACE: Reset environment")
    print(f"   R: Toggle auto-reset")
    print(f"   Click the window and use keys!")
    
    # Initialize pygame for event handling
    pygame.init()
    clock = pygame.time.Clock()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Reset environment
    obs, info = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    total_reward = 0.0
    running = True
    auto_reset = True
    
    print(f"\n🚀 Starting evaluation...")
    
    episode_rewards = []
    episode_lengths = []
    
    while running and (args.num_episodes == 0 or episode_count < args.num_episodes):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    print(f"🔄 Manual reset triggered")
                    obs, info = env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                elif event.key == pygame.K_r:
                    auto_reset = not auto_reset
                    print(f"🔄 Auto-reset: {'ON' if auto_reset else 'OFF'}")
        
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs_dict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[1], device=device)
        
        # Get action from policy
        with torch.no_grad():
            actions = policy.act(obs_dict, deterministic=True)  # Use deterministic for evaluation
        
        # Convert action to numpy and remove batch dimension
        action = actions.cpu().numpy()[0]
        
        # Apply action processing (matching training settings)
        if args.action_scale != 1.0:
            action *= args.action_scale
        if args.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update episode tracking
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        # Print step info (every 50 steps to avoid spam)
        if episode_length % 50 == 0:
            print(f"Step {episode_length:3d}: Reward {reward:6.3f}, Episode Reward: {episode_reward:8.3f}")
        
        # Handle episode completion
        if done:
            episode_count += 1
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            avg_reward = total_reward / episode_count
            
            print(f"\n📊 Episode {episode_count} Complete!")
            print(f"   Reward: {episode_reward:8.3f}")
            print(f"   Length: {episode_length:3d} steps")
            print(f"   Average Reward: {avg_reward:8.3f}")
            if len(episode_rewards) > 1:
                print(f"   Best Reward: {max(episode_rewards):8.3f}")
                print(f"   Worst Reward: {min(episode_rewards):8.3f}")
                print(f"   Reward Std: {np.std(episode_rewards):8.3f}")
            
            # Goal achievement check (sparse reward > 0 indicates goal reached)
            goal_reached = reward > 0 or episode_reward > 0.5  # Adjust threshold as needed
            if goal_reached:
                print(f"🎯 GOAL REACHED! 🎉")
            
            # Reset for next episode
            if auto_reset:
                obs, info = env.reset()
            else:
                print(f"⏸️  Auto-reset disabled. Press SPACE to reset manually.")
                # Keep current state until manual reset
            
            episode_reward = 0.0
            episode_length = 0
        
        # Control frame rate
        clock.tick(args.fps)
    
    # Final statistics
    if episode_rewards:
        print(f"\n📊 Final Statistics ({len(episode_rewards)} episodes):")
        print(f"   Average Reward: {np.mean(episode_rewards):8.3f} ± {np.std(episode_rewards):6.3f}")
        print(f"   Best Reward: {np.max(episode_rewards):8.3f}")
        print(f"   Worst Reward: {np.min(episode_rewards):8.3f}")
        print(f"   Average Length: {np.mean(episode_lengths):6.1f} ± {np.std(episode_lengths):4.1f}")
        
        # Success rate (episodes with positive reward)
        successful_episodes = sum(1 for r in episode_rewards if r > 0.5)
        success_rate = successful_episodes / len(episode_rewards) * 100
        print(f"   Success Rate: {successful_episodes}/{len(episode_rewards)} ({success_rate:.1f}%)")
    
    print(f"\n✅ Evaluation completed!")

def main():
    """Main evaluation function."""
    args = get_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Interactive Policy Evaluation")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    try:
        # Create environment
        env = create_env(args.env_name, args)
        
        # Load trained policy
        policy = load_trained_policy(args.model_path, env, device)
        
        # Run interactive evaluation
        run_interactive_evaluation(policy, env, args, device)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        try:
            env.close()
        except:
            pass
        pygame.quit()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        pygame.quit()
        sys.exit(0)
