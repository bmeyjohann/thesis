#!/usr/bin/env python3
"""
Test script to examine enhanced OGBench observations
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'fasttd3', 'fast_sac'))

import ogbench
import torch
from environments.ogbench_env import OGBenchEnv

def test_obs_config(config_name, obs_config):
    print(f"\n=== Testing {config_name} ===")
    print(f"Config: {obs_config}")
    
    env = OGBenchEnv(
        env_name='pointmaze-medium-v0',
        num_envs=1,
        obs_config=obs_config
    )
    
    print(f"Calculated obs dimension: {env.num_obs}")
    
    # Reset and examine initial observation
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Take a step and examine
    action = torch.randn(1, env.num_actions)
    obs, reward, done, info = env.step(action)
    print(f"After step - Observation: {obs}")
    print(f"Reward: {reward}")
    
    return env.num_obs

# Test different configurations
configs = {
    "Full Enhanced": {
        'include_goal': True,
        'include_goal_distance': True,
        'include_velocity': True,
        'include_goal_direction': True,
        'normalize_positions': False,
    },
    "Goal Only": {
        'include_goal': True,
        'include_goal_distance': False,
        'include_velocity': False,
        'include_goal_direction': False,
        'normalize_positions': False,
    },
    "Minimal + Distance": {
        'include_goal': True,
        'include_goal_distance': True,
        'include_velocity': False,
        'include_goal_direction': False,
        'normalize_positions': False,
    },
    "Normalized Full": {
        'include_goal': True,
        'include_goal_distance': True,
        'include_velocity': True,
        'include_goal_direction': True,
        'normalize_positions': True,
    }
}

print("Testing Enhanced OGBench Observations")
print("====================================")

for config_name, obs_config in configs.items():
    obs_dim = test_obs_config(config_name, obs_config)
    
print(f"\nSummary:")
print(f"Base observation: [agent_x, agent_y] = 2 dims")
print(f"+ goal: [goal_x, goal_y] = +2 dims")
print(f"+ distance: [euclidean_distance] = +1 dim")
print(f"+ direction: [norm_dx, norm_dy] = +2 dims")
print(f"+ velocity: [vx, vy] = +2 dims (if available)")
print(f"Total possible: 2 + 2 + 1 + 2 + 2 = 9 dimensions")