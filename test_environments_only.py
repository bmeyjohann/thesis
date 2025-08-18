#!/usr/bin/env python3
"""
Test script for environment setups only (no RSL-RL required)
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'fasttd3', 'fast_sac'))

import torch
import numpy as np
import gymnasium as gym
import ogbench
from simple_dynamic_pointmaze import SimpleDynamicPointMaze
from environments.ogbench_env import OGBenchEnv

# Register the simple environment
gym.register(
    id='SimpleDynamicPointMaze-v0',
    entry_point='simple_dynamic_pointmaze:SimpleDynamicPointMaze',
    max_episode_steps=500,
)

def test_simple_env():
    """Test SimpleDynamicPointMaze environment."""
    print("\n🔵 Testing SimpleDynamicPointMaze...")
    
    env = SimpleDynamicPointMaze(arena_size=10.0, max_episode_steps=100)
    
    obs, info = env.reset()
    print(f"   ✅ Reset successful")
    print(f"   📐 Obs shape: {obs.shape}, Goal: {info['goal']}")
    
    # Test enhanced observation creation
    agent_pos = obs[:2]
    goal_pos = info['goal']
    distance = np.linalg.norm(goal_pos - agent_pos)
    direction = goal_pos - agent_pos
    direction = direction / np.linalg.norm(direction)
    
    enhanced_obs = np.concatenate([agent_pos, goal_pos, [distance], direction])
    print(f"   📊 Enhanced obs shape: {enhanced_obs.shape} (should be 7)")
    print(f"   🎯 Enhanced obs: {enhanced_obs}")
    
    # Test step with dense rewards
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   🏆 Dense reward: {reward:.4f}")
    
    env.close()
    return True

def test_ogbench_enhanced():
    """Test OGBench environment with enhanced observations."""
    print("\n🔴 Testing OGBench with Enhanced Observations...")
    
    # Test different observation configurations
    configs = [
        {'include_goal': True, 'include_goal_distance': True, 'include_goal_direction': True},
        {'include_goal': True, 'include_goal_distance': False, 'include_goal_direction': False},
    ]
    
    for i, obs_config in enumerate(configs):
        print(f"\n   Config {i+1}: {obs_config}")
        
        env = OGBenchEnv(
            env_name='pointmaze-medium-v0',
            num_envs=1,
            obs_config=obs_config
        )
        
        print(f"   📐 Calculated obs dim: {env.num_obs}")
        
        obs = env.reset()
        print(f"   ✅ Reset successful, obs shape: {obs.shape}")
        print(f"   📊 Sample obs: {obs[0]}")
        
        actions = torch.randn(1, env.num_actions)
        obs, rewards, dones, infos = env.step(actions)
        print(f"   ✅ Step successful, reward: {rewards[0]:.4f}")
        
    return True

def test_reward_types():
    """Test different reward configurations."""
    print("\n🟡 Testing Reward Types...")
    
    # Test simple environment with sparse vs dense
    print("   SimpleDynamicPointMaze:")
    
    # Dense rewards (default)
    env_dense = SimpleDynamicPointMaze(distance_reward_scale=0.01)
    obs, info = env_dense.reset()
    
    for i in range(3):
        action = env_dense.action_space.sample()
        obs, reward, terminated, truncated, info = env_dense.step(action)
        print(f"     Step {i+1} dense reward: {reward:.4f}")
    
    env_dense.close()
    
    # Sparse rewards (goal only)  
    env_sparse = SimpleDynamicPointMaze(distance_reward_scale=0.0, goal_reward=1.0)
    obs, info = env_sparse.reset()
    
    for i in range(3):
        action = env_sparse.action_space.sample()
        obs, reward, terminated, truncated, info = env_sparse.step(action)
        print(f"     Step {i+1} sparse reward: {reward:.4f}")
        if terminated:
            print("       🎯 Goal reached!")
            break
    
    env_sparse.close()
    return True

def main():
    print("🧪 Testing Environment Setups (No RSL-RL)")
    print("=" * 45)
    
    try:
        success1 = test_simple_env()
        success2 = test_ogbench_enhanced()  
        success3 = test_reward_types()
        
        if success1 and success2 and success3:
            print(f"\n🎉 All environment tests passed!")
            print(f"✅ Ready for cluster training with RSL-RL")
            print(f"\n🚀 To run on cluster:")
            print(f"   ./submit_job.sh goal_navigation_experiments.sbatch")
        else:
            print(f"\n⚠️  Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()