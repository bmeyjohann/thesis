#!/usr/bin/env python3
"""
Test script to examine what's currently in OGBench observations
"""

import ogbench
import gymnasium as gym
import numpy as np

# Test a simple pointmaze environment
env_name = 'pointmaze-medium-v0'
env = gym.make(env_name)

print(f"Testing {env_name}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Reset and examine initial observation
obs, info = env.reset()
print(f"\nObservation shape: {obs.shape}")
print(f"Observation: {obs}")
print(f"\nInfo keys: {list(info.keys()) if info else 'None'}")
print(f"Info content: {info}")

# Take a step and examine
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"\nAfter step:")
print(f"Observation: {obs}")
print(f"Reward: {reward}")
print(f"Info keys: {list(info.keys()) if info else 'None'}")
print(f"Info content: {info}")

env.close()