#!/usr/bin/env python3
"""
Debug teleoperation script for OGBench PointMaze environments.
This script provides extensive debug logging of observations, info, and reward calculations
while allowing manual control of the agent for understanding the coordinate system and environment behavior.
"""

import time
import pygame
import gymnasium as gym
import numpy as np
import os

# Fix WSL window positioning issues
os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center windows

# Import ogbench to register environments
import ogbench

# Import wrappers from the proper locations
from ogbench.wrappers import SpeedWrapper

class DebugControlWindow:
    """Separate control window for keyboard input and debug information."""
    
    def __init__(self, width=500, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🔍 Debug Teleoperation Control Panel")
        
        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)
        
        # Font
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        self.quit_requested = False
        
    def update(self, step_count, obs=None, reward_components=None, goal_info=None):
        """Update the control window display."""
        self.screen.fill(self.black)
        
        # Title
        title = self.large_font.render("🔍 DEBUG CONTROL", True, self.yellow)
        self.screen.blit(title, (10, 10))
        
        # Step counter
        step_text = self.font.render(f"Step: {step_count}", True, self.white)
        self.screen.blit(step_text, (10, 50))
        
        # Controls section
        controls_y = 80
        controls = [
            "KEYBOARD CONTROLS:",
            "↑↓←→ Arrow Keys: Move agent",
            "ESC: Exit debug session",
            "",
            "FOCUS THIS WINDOW for keyboard control!"
        ]
        
        for i, control in enumerate(controls):
            if i == 0:
                color = self.yellow
                font = self.font
            elif "FOCUS" in control:
                color = self.green
                font = self.font
            else:
                color = self.white
                font = self.small_font
            
            text = font.render(control, True, color)
            self.screen.blit(text, (10, controls_y + i * 25))
        
        # Current observation info (if available)
        if obs is not None:
            obs_y = 200
            obs_title = self.font.render("CURRENT STATE:", True, self.blue)
            self.screen.blit(obs_title, (10, obs_y))
            
            if hasattr(obs, '__len__') and len(obs) >= 2:
                pos_text = self.small_font.render(f"Position: ({obs[0]:.3f}, {obs[1]:.3f})", True, self.white)
                self.screen.blit(pos_text, (10, obs_y + 25))
                
                if len(obs) == 4:
                    rel_goal_text = self.small_font.render(f"Rel Goal: ({obs[2]:.3f}, {obs[3]:.3f})", True, self.white)
                    self.screen.blit(rel_goal_text, (10, obs_y + 45))
                    
                    distance = np.linalg.norm(obs[2:4])
                    dist_text = self.small_font.render(f"Distance: {distance:.3f}", True, self.white)
                    self.screen.blit(dist_text, (10, obs_y + 65))
        
        # Reward components (if available)
        if reward_components:
            reward_y = 290
            reward_title = self.font.render("REWARDS:", True, self.green)
            self.screen.blit(reward_title, (10, reward_y))
            
            y_offset = 315
            for key, value in reward_components.items():
                color = self.green if value > 0 else self.red if value < 0 else self.white
                reward_text = self.small_font.render(f"{key}: {value:.4f}", True, color)
                self.screen.blit(reward_text, (10, y_offset))
                y_offset += 18
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events and return keyboard input."""
        keys_pressed = pygame.key.get_pressed()
        action = np.array([0.0, 0.0])
        
        # Handle discrete key presses for movement
        if keys_pressed[pygame.K_UP]:
            action[1] = 1.0
        if keys_pressed[pygame.K_DOWN]:
            action[1] = -1.0
        if keys_pressed[pygame.K_LEFT]:
            action[0] = -1.0
        if keys_pressed[pygame.K_RIGHT]:
            action[0] = 1.0
        
        # Handle other events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
        
        return action, self.quit_requested
    
    def close(self):
        """Close the control window."""
        pygame.quit()

# Reward calculation functions from training script
def calculate_distance_reward(obs, info, enable_shaping=True, prev_distance=None):
    """Calculate distance-based reward to help with sparse rewards."""
    if not enable_shaping:
        return 0.0, None
        
    try:
        obs_array = obs if isinstance(obs, np.ndarray) else np.array(obs)
        
        # Check if observation is goal-conditioned (4D: [agent_x, agent_y, relative_goal_x, relative_goal_y])
        if len(obs_array) == 4:
            # Extract relative goal position (last 2 elements)
            relative_goal = obs_array[2:4]
            # Distance to goal is the magnitude of the relative goal vector
            current_distance = np.linalg.norm(relative_goal)
        else:
            # Fallback: try to get goal from info (original method)
            goal_pos = info.get('goal', None)
            if goal_pos is None:
                return 0.0, None
            
            current_pos = obs_array[:2] if len(obs_array) >= 2 else obs_array
            goal_pos = np.array(goal_pos) if not isinstance(goal_pos, np.ndarray) else goal_pos
            current_distance = np.linalg.norm(current_pos[:2] - goal_pos[:2])
        
        # Calculate distance-based reward
        max_distance = 30.0
        
        # Base reward for being close (small component)
        normalized_distance = min(current_distance / max_distance, 1.0)
        closeness_reward = (1.0 - normalized_distance) * 0.2
        
        # Main reward for improving distance (if we have previous distance)
        improvement_reward = 0.0
        if prev_distance is not None:
            distance_change = prev_distance - current_distance  # Positive = getting closer
            improvement_reward = distance_change * 1.0
        
        # Bonus for being very close
        bonus = 0.5 if current_distance < 2.0 else 0.0
        
        total_reward = closeness_reward + improvement_reward + bonus
        
        return total_reward, current_distance
        
    except Exception as e:
        print(f"Error calculating distance reward: {e}")
        return 0.0, None

def calculate_directional_reward(obs, action, enable_shaping=True):
    """Calculate reward based on whether action moves toward or away from goal."""
    if not enable_shaping:
        return 0.0
        
    try:
        obs_array = obs if isinstance(obs, np.ndarray) else np.array(obs)
        action_array = action if isinstance(action, np.ndarray) else np.array(action)
        
        # Only works with goal-conditioned observations
        if len(obs_array) != 4:
            return 0.0
            
        # Extract relative goal position (last 2 elements)
        relative_goal = obs_array[2:4]
        
        # Normalize the relative goal to get desired direction
        goal_distance = np.linalg.norm(relative_goal)
        if goal_distance < 0.01:  # Very close to goal
            return 0.5  # Small bonus for being at goal
            
        desired_direction = relative_goal / goal_distance
        
        # Normalize action to get actual direction
        action_magnitude = np.linalg.norm(action_array)
        if action_magnitude < 0.01:  # No movement
            return 0.0
            
        action_direction = action_array / action_magnitude
        
        # Calculate dot product to measure alignment
        alignment = np.dot(desired_direction, action_direction)
        
        # Convert alignment to reward
        directional_reward = alignment * 2.0
        
        return directional_reward
        
    except Exception as e:
        print(f"Error calculating directional reward: {e}")
        return 0.0

def debug_full_observation_and_info(obs, info, step_count, action=None, reward_components=None):
    """Comprehensive debug function to understand the environment state."""
    print(f"\n{'='*80}")
    print(f"🔍 DEBUG STEP {step_count}")
    print(f"{'='*80}")
    
    # Observation analysis
    print("📊 OBSERVATION ANALYSIS:")
    print(f"   Type: {type(obs)}")
    print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
    print(f"   Value: {obs}")
    print(f"   Data type: {obs.dtype if hasattr(obs, 'dtype') else type(obs)}")
    
    if hasattr(obs, '__len__') and len(obs) >= 2:
        print(f"   Agent position: [{obs[0]:.3f}, {obs[1]:.3f}]")
        if len(obs) == 4:
            print(f"   Additional data: [{obs[2]:.3f}, {obs[3]:.3f}]")
            print(f"   → Likely relative goal: [{obs[2]:.3f}, {obs[3]:.3f}]")
            distance_to_goal = np.linalg.norm(obs[2:4])
            print(f"   → Distance to goal: {distance_to_goal:.3f}")
    
    # Info analysis
    print("\n📋 INFO DICTIONARY ANALYSIS:")
    if isinstance(info, dict):
        print(f"   Keys: {list(info.keys())}")
        for key, value in info.items():
            print(f"   {key}: {value}")
            
            # Special analysis for key fields
            if key == 'goal' and value is not None:
                goal_array = np.array(value)
                print(f"      → Goal position: [{goal_array[0]:.3f}, {goal_array[1]:.3f}]")
                if hasattr(obs, '__len__') and len(obs) >= 2:
                    agent_pos = obs[:2]
                    distance = np.linalg.norm(agent_pos - goal_array)
                    print(f"      → Distance from agent: {distance:.3f}")
    else:
        print(f"   Info is not a dict: {type(info)} = {info}")
    
    # Action analysis
    if action is not None:
        print(f"\n🎮 ACTION ANALYSIS:")
        print(f"   Type: {type(action)}")
        print(f"   Value: {action}")
        if hasattr(action, '__len__') and len(action) >= 2:
            direction_x = "RIGHT" if action[0] > 0.1 else "LEFT" if action[0] < -0.1 else "NONE"
            direction_y = "UP" if action[1] > 0.1 else "DOWN" if action[1] < -0.1 else "NONE"
            print(f"   Direction: X={direction_x}, Y={direction_y}")
            magnitude = np.linalg.norm(action)
            print(f"   Magnitude: {magnitude:.3f}")
    
    # Reward analysis
    if reward_components is not None:
        print(f"\n💰 REWARD COMPONENTS:")
        for key, value in reward_components.items():
            print(f"   {key}: {value:.4f}")
    
    # Coordinate system help
    if hasattr(obs, '__len__') and len(obs) >= 2:
        print(f"\n🗺️  COORDINATE SYSTEM REFERENCE:")
        print(f"   Current position: ({obs[0]:.3f}, {obs[1]:.3f})")
        print(f"   → Moving RIGHT increases X")
        print(f"   → Moving UP increases Y")
        print(f"   → Origin (0,0) is typically at maze center/corner")
        
        if len(obs) == 4:
            rel_goal = obs[2:4]
            optimal_x = "RIGHT" if rel_goal[0] > 0 else "LEFT" if rel_goal[0] < 0 else "NONE"
            optimal_y = "UP" if rel_goal[1] > 0 else "DOWN" if rel_goal[1] < 0 else "NONE"
            print(f"   To reach goal, should move: {optimal_x} and {optimal_y}")
    
    print(f"{'='*80}\n")

def main():
    """
    Main function to run debug teleoperation on a point maze environment.
    """
    # List of available point maze environments
    available_envs = [
        'pointmaze-medium-v0',
        'pointmaze-large-v0', 
        'pointmaze-giant-v0',
        'pointmaze-teleport-v0'
    ]
    
    print("🔍 === OGBench Point Maze DEBUG Teleoperation ===")
    print("This script provides extensive debug logging to understand:")
    print("  • Coordinate system and observation format")
    print("  • Goal information and environment state")
    print("  • Reward calculations and components")
    print("  • Action effects and movement patterns")
    
    # Use visual rendering mode
    render_mode = "human"
    print(f"\n🎯 Starting in visual mode with debug logging...")
    
    print("\nAvailable environments:")
    for i, env_name in enumerate(available_envs):
        print(f"  {i+1}. {env_name}")
    
    # Let user choose environment
    while True:
        try:
            choice = int(input(f"\nSelect environment (1-{len(available_envs)}): ")) - 1
            if 0 <= choice < len(available_envs):
                selected_env = available_envs[choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nStarting {selected_env} with debug logging...")
    
    # Create the environment with larger window
    try:
        env = gym.make(
            selected_env,
            render_mode=render_mode,
            max_episode_steps=1000,
            width=800,
            height=600
        )
        
        # Apply speed wrapper for responsive movement
        env = SpeedWrapper(env, speed_multiplier=3.0)
        
    except Exception as e:
        print(f"Failed to create environment: {e}")
        print("This might be a MuJoCo/OpenGL issue. Try installing proper graphics drivers.")
        return

    # Create the control window outside try block so it's accessible in finally
    control_window = None
    
    try:
        # Create the separate control window for keyboard input
        control_window = DebugControlWindow()
        
        print("\n🔍 === DEBUG TELEOPERATION MODE ===")
        print("🎮 TWO WINDOWS WILL OPEN:")
        print("   1. Environment visualization window (MuJoCo)")  
        print("   2. Debug control window (focus for keyboard input)")
        print("🔍 Each step will show detailed debug information")
        print("💰 Reward calculations will be displayed")
        print("🗺️  Coordinate system will be explained")
        print("❌ Focus the CONTROL WINDOW and use arrow keys to move")
        print("❌ Press ESC in control window to exit.\n")
        
        obs, info = env.reset()
        
        # Initial debug output
        print("🏁 INITIAL STATE:")
        debug_full_observation_and_info(obs, info, 0)
        
        running = True
        step_count = 0
        prev_distance = None
        terminated = False
        truncated = False
        
        # Initialize distance tracking
        if hasattr(obs, '__len__') and len(obs) == 4:
            prev_distance = np.linalg.norm(obs[2:4])
        elif isinstance(info, dict) and 'goal' in info:
            goal_pos = np.array(info['goal'])
            agent_pos = obs[:2] if hasattr(obs, '__len__') else np.array([0, 0])
            prev_distance = np.linalg.norm(agent_pos - goal_pos)
        
        while running:
            # Get action from control window
            action_taken, quit_requested = control_window.handle_events()
            
            if quit_requested:
                running = False
                break
            
            # Only step if there's an action (to avoid unnecessary steps)
            if np.any(action_taken != 0):
                obs, reward, terminated, truncated, info = env.step(action_taken)
                step_count += 1
                
                # Calculate reward components for debugging
                distance_reward, current_distance = calculate_distance_reward(obs, info, True, prev_distance)
                directional_reward = calculate_directional_reward(obs, action_taken, True)
                
                reward_components = {
                    'sparse_reward': reward,
                    'distance_reward': distance_reward,
                    'directional_reward': directional_reward,
                    'total_reward': reward + distance_reward + directional_reward
                }
                
                # Update previous distance
                prev_distance = current_distance
                
                # Full debug output for each step
                debug_full_observation_and_info(obs, info, step_count, action_taken, reward_components)
                
                # Update control window with current state
                control_window.update(step_count, obs, reward_components)
            else:
                # Update control window even without movement
                control_window.update(step_count, obs)
            
            # Check for episode termination
            if terminated or truncated:
                print(f"\n🏁 EPISODE FINISHED after {step_count} steps!")
                
                if terminated:
                    print("✅ Goal reached! 🎉")
                else:
                    print("⏰ Episode truncated (time limit reached)")
                    
                print("🔄 Resetting environment...")
                time.sleep(2)
                obs, info = env.reset()
                step_count = 0
                prev_distance = None
                
                # Reset distance tracking
                if hasattr(obs, '__len__') and len(obs) == 4:
                    prev_distance = np.linalg.norm(obs[2:4])
                elif isinstance(info, dict) and 'goal' in info:
                    goal_pos = np.array(info['goal'])
                    agent_pos = obs[:2] if hasattr(obs, '__len__') else np.array([0, 0])
                    prev_distance = np.linalg.norm(agent_pos - goal_pos)
                
                print("🏁 NEW EPISODE STATE:")
                debug_full_observation_and_info(obs, info, 0)

            # Moderate delay for readability
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n👋 Exiting debug session...")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("This might be related to MuJoCo/OpenGL setup.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check MuJoCo installation: python -c 'import mujoco; print(mujoco.__version__)'")
        print("2. Install missing graphics libraries")
        print("3. Try running the original teleoperation_example.py first")
    finally:
        print("🔚 Closing environment and control window.")
        try:
            env.close()
        except:
            pass
        if control_window is not None:
            try:
                control_window.close()
            except:
                pass


if __name__ == "__main__":
    main() 