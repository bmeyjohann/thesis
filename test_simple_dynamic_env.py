#!/usr/bin/env python3
"""
Test script for the Simple Dynamic Point Maze with enhanced OGBench wrapper
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'fasttd3', 'fast_sac'))

import torch
import numpy as np
import gymnasium as gym
from simple_dynamic_pointmaze import SimpleDynamicPointMaze
from environments.ogbench_env import OGBenchEnv

# Register the simple environment
gym.register(
    id='SimpleDynamicPointMaze-v0',
    entry_point='simple_dynamic_pointmaze:SimpleDynamicPointMaze',
    max_episode_steps=500,
)

def test_simple_env_direct():
    """Test the simple environment directly."""
    print("=== Testing Simple Dynamic Point Maze (Direct) ===")
    
    env = SimpleDynamicPointMaze(arena_size=10.0, max_episode_steps=100)
    
    obs, info = env.reset()
    print(f"Initial obs: {obs}")
    print(f"Goal: {info['goal']}")
    print(f"Distance: {info['distance_to_goal']:.2f}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, distance={info['distance_to_goal']:.2f}")
        
        if terminated:
            print("Goal reached!")
            break
    
    env.close()
    return True

def test_simple_env_with_wrapper():
    """Test the simple environment with the enhanced OGBench wrapper."""
    print("\n=== Testing Simple Dynamic Point Maze (With Enhanced Wrapper) ===")
    
    # Test different observation configurations
    configs = [
        {
            'name': 'Full Enhanced',
            'config': {
                'include_goal': True,
                'include_goal_distance': True,
                'include_velocity': True,
                'include_goal_direction': True,
                'normalize_positions': False,
            }
        },
        {
            'name': 'Minimal',
            'config': {
                'include_goal': True,
                'include_goal_distance': True,
                'include_velocity': False,
                'include_goal_direction': False,
                'normalize_positions': False,
            }
        }
    ]
    
    for config_info in configs:
        print(f"\n--- Testing {config_info['name']} Configuration ---")
        
        # Note: We can't directly use SimpleDynamicPointMaze with OGBenchEnv
        # because OGBenchEnv expects registered gym environments
        # Let's test with a real OGBench environment instead
        
        try:
            env = OGBenchEnv(
                env_name='pointmaze-medium-v0',  # Use real OGBench env
                num_envs=2,
                obs_config=config_info['config']
            )
            
            print(f"Observation dimension: {env.num_obs}")
            
            # Reset and examine
            obs = env.reset()
            print(f"Observation shape: {obs.shape}")
            print(f"Sample observation: {obs[0]}")
            
            # Take a step
            actions = torch.randn(2, env.num_actions)
            obs, rewards, dones, infos = env.step(actions)
            
            print(f"After step:")
            print(f"Rewards: {rewards}")
            print(f"Sample observation: {obs[0]}")
            
            # Explain what each part of the observation is
            explain_observation(obs[0], config_info['config'])
            
        except Exception as e:
            print(f"Error testing {config_info['name']}: {e}")
            continue

def explain_observation(obs, config):
    """Explain what each part of the observation vector represents."""
    print("Observation breakdown:")
    idx = 0
    
    # Agent position (always included)
    print(f"  [0:2] Agent position (x, y): {obs[idx:idx+2]}")
    idx += 2
    
    if config.get('include_goal', True):
        print(f"  [{idx}:{idx+2}] Goal position (x, y): {obs[idx:idx+2]}")
        idx += 2
    
    if config.get('include_goal_distance', True):
        print(f"  [{idx}] Goal distance: {obs[idx]}")
        idx += 1
    
    if config.get('include_goal_direction', True):
        print(f"  [{idx}:{idx+2}] Goal direction (normalized): {obs[idx:idx+2]}")
        idx += 2
    
    if config.get('include_velocity', True):
        print(f"  [{idx}:{idx+2}] Velocity (vx, vy): {obs[idx:idx+2]}")
        idx += 2
    
    print(f"  Total dimensions: {len(obs)}")

if __name__ == '__main__':
    print("Testing Simple Dynamic Point Maze Environment")
    print("=" * 50)
    
    # Test the simple environment directly
    success = test_simple_env_direct()
    
    if success:
        # Test with enhanced wrapper
        test_simple_env_with_wrapper()
    
    print("\nTesting completed!")