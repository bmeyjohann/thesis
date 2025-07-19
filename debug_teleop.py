#!/usr/bin/env python3
"""
Correct teleoperation debug script for OGBench PointMaze environments.

This script demonstrates the proper use of wrappers to achieve teleoperation
with a separate control window and detailed console debug output.

It follows a clean separation of concerns:
- `RelativeGoalWrapper`: Injects goal information into observations.
- `SpeedWrapper`: Makes agent movement responsive.
- `DirectTeleopWrapper`: Uses a teleop interface for direct, non-blocking control.
"""

import time
import pygame
import gymnasium as gym
import numpy as np
import os

# Fix WSL window positioning issues
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Import ogbench to register environments
import ogbench

# Import the necessary, existing wrappers and the new teleop interface
from ogbench.wrappers import (
    DirectTeleopWrapper,
    RelativeGoalWrapper,
    SpeedWrapper
)
from ogbench.teleop import ControlWindowTeleop

# --- Reward calculation functions (from train_pointmaze_dynamic.py) ---

def calculate_distance_reward(obs, prev_distance=None):
    """
    Calculates distance-based reward from a goal-conditioned observation.
    Assumes observation is [agent_pos, relative_goal_pos].
    """
    if len(obs) != 4:
        return 0.0, None  # Cannot calculate without relative goal

    relative_goal = obs[2:4]
    current_distance = np.linalg.norm(relative_goal)
    
    # Reward for improving distance
    improvement_reward = 0.0
    if prev_distance is not None:
        distance_change = prev_distance - current_distance
        improvement_reward = distance_change * 1.0
        
    # Bonus for being very close
    bonus = 0.5 if current_distance < 2.0 else 0.0
    
    return improvement_reward + bonus, current_distance

def calculate_directional_reward(obs, action):
    """
    Calculates reward based on how well the action aligns with the goal direction.
    Assumes observation is [agent_pos, relative_goal_pos].
    """
    if len(obs) != 4 or np.linalg.norm(action) < 0.01:
        return 0.0

    relative_goal = obs[2:4]
    goal_distance = np.linalg.norm(relative_goal)
    if goal_distance < 0.01:
        return 0.5 # At goal bonus

    desired_direction = relative_goal / goal_distance
    action_direction = action / np.linalg.norm(action)
    
    alignment = np.dot(desired_direction, action_direction)
    return alignment * 2.0

# --- Main Debug Logic ---

def main():
    """Main function to run the debug teleoperation."""
    
    available_envs = [
        'pointmaze-medium-v0', 'pointmaze-large-v0', 
        'pointmaze-giant-v0', 'pointmaze-teleport-v0'
    ]
    
    print("🤖 === Correct Debug Teleoperation Script ===")
    print("This script uses your existing wrappers correctly.")
    
    print("\nAvailable environments:")
    for i, env_name in enumerate(available_envs):
        print(f"  {i+1}. {env_name}")
    
    # Get user input
    choice = int(input(f"\nSelect environment (1-{len(available_envs)}): ")) - 1
    selected_env = available_envs[choice]

    print(f"\n🚀 Launching {selected_env}...")

    # Create the teleoperation interface first
    teleop_interface = ControlWindowTeleop(show_debug_info=False)

    try:
        # --- Wrapper Composition ---
        # 1. Create the base environment
        env = gym.make(selected_env, render_mode="human", max_episode_steps=1000)
        
        # 2. IMPORTANT: Add goal information to observations
        env = RelativeGoalWrapper(env)
        
        # 3. Apply speed enhancement
        env = SpeedWrapper(env, speed_multiplier=3.0)
        
        # 4. Add the direct teleoperation wrapper
        env = DirectTeleopWrapper(env, teleop_interface)
        
        print("\n✅ Environment setup complete with proper wrappers.")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        teleop_interface.close()
        return

    try:
        obs, info = env.reset()
        prev_distance = np.linalg.norm(obs[2:4])
        step_count = 0
        
        while not teleop_interface.should_quit():
            # The DirectTeleopWrapper handles getting the action from the window.
            # We pass a dummy action (None) which is ignored by the wrapper.
            obs, reward, terminated, truncated, info = env.step(None)
            step_count += 1
            
            # Update the teleop window with the latest state info
            teleop_interface.update_state(obs, info, step_count)

            # --- Debug Console Output ---
            # We only print details if the action was non-zero to avoid spam
            action = info.get("human_action", np.array([0.0, 0.0]))
            if np.any(action != 0):
                print(f"\n{'='*40} Step {step_count} {'='*40}")
                
                # Observation
                print(f"👀 Observation (shape: {obs.shape}):")
                print(f"   Agent Pos:  [{obs[0]:.3f}, {obs[1]:.3f}]")
                print(f"   Rel. Goal:  [{obs[2]:.3f}, {obs[3]:.3f}]")

                # Rewards
                dist_reward, current_distance = calculate_distance_reward(obs, prev_distance)
                dir_reward = calculate_directional_reward(obs, action)
                prev_distance = current_distance
                print("\n💰 Rewards:")
                print(f"   - Environment reward: {reward:.3f}")
                print(f"   - Distance reward:    {dist_reward:.3f}")
                print(f"   - Directional reward: {dir_reward:.3f}")
                print(f"   - TOTAL shaped reward: {(reward + dist_reward + dir_reward):.3f}")
                
                # Info dictionary
                print(f"\nℹ️  Info dict keys: {list(info.keys())}")
                if 'goal' in info:
                    print(f"   Absolute Goal: [{info['goal'][0]:.2f}, {info['goal'][1]:.2f}]")

            if terminated or truncated:
                print("\n🏁 Episode Finished!")
                obs, info = env.reset()
                prev_distance = np.linalg.norm(obs[2:4])
                step_count = 0

            time.sleep(0.02) # Small delay

    except KeyboardInterrupt:
        print("\n👋 Exiting via Ctrl+C...")
    finally:
        print("🔚 Closing environment.")
        env.close()

if __name__ == "__main__":
    main()
    print("\n✅ Debug script finished.") 