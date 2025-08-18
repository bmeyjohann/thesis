#!/usr/bin/env python3
"""
Test script for goal navigation experiments setup
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'fasttd3', 'fast_sac'))

import torch
import gymnasium as gym
from simple_dynamic_pointmaze import SimpleDynamicPointMaze

# Register the simple environment
gym.register(
    id='SimpleDynamicPointMaze-v0',
    entry_point='simple_dynamic_pointmaze:SimpleDynamicPointMaze',
    max_episode_steps=500,
)

def test_experiment_config(env_type, reward_type):
    """Test a single experiment configuration."""
    print(f"\n🧪 Testing {env_type} + {reward_type}")
    
    try:
        # Import here to avoid circular imports
        from train_goal_navigation_experiments import UnifiedRSLRLVecEnv
        
        # Create environment
        env = UnifiedRSLRLVecEnv(
            env_type=env_type,
            reward_type=reward_type,
            num_envs=2  # Small number for testing
        )
        
        print(f"   ✅ Environment created successfully")
        print(f"   📐 Obs dim: {env.num_obs}, Action dim: {env.num_actions}")
        
        # Test reset
        obs = env.reset()
        print(f"   ✅ Reset successful, obs shape: {obs['policy'].shape}")
        
        # Test step
        actions = torch.randn(env.num_envs, env.num_actions)
        obs, rewards, dones, infos = env.step(actions)
        
        print(f"   ✅ Step successful")
        print(f"   🎯 Sample obs: {obs['policy'][0][:5]}...")  # First 5 elements
        print(f"   🏆 Rewards: {rewards}")
        print(f"   ✅ {env_type} + {reward_type} configuration working!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🔬 Testing Goal Navigation Experiments Setup")
    print("=" * 50)
    
    # Test all four configurations
    configs = [
        ('simple', 'sparse'),
        ('simple', 'dense'), 
        ('ogbench', 'sparse'),
        ('ogbench', 'dense'),
    ]
    
    results = {}
    for env_type, reward_type in configs:
        results[f"{env_type}_{reward_type}"] = test_experiment_config(env_type, reward_type)
    
    print(f"\n📊 Test Results Summary:")
    print("=" * 30)
    
    all_passed = True
    for config_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {config_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All experiment configurations are working!")
        print(f"🚀 Ready to submit to cluster with:")
        print(f"   ./submit_job.sh goal_navigation_experiments.sbatch")
    else:
        print(f"\n⚠️  Some configurations failed. Please fix before submitting to cluster.")

if __name__ == '__main__':
    main()