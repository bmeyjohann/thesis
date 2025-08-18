#!/usr/bin/env python3
"""
Simple Dynamic Point Maze Environment

This creates a simplified point maze environment with:
- Dynamic goal positions that change each episode
- Dynamic spawn positions
- No walls/obstacles (open space navigation)
- Dense euclidean distance reward
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt


class SimpleDynamicPointMaze(gym.Env):
    """
    A simple 2D point navigation environment with dynamic goals and spawn points.
    
    The agent must navigate to randomly placed goals in an open 2D space.
    """
    
    def __init__(self, 
                 arena_size: float = 20.0,
                 max_episode_steps: int = 500,
                 goal_reward: float = 1.0,
                 distance_reward_scale: float = 0.01,
                 action_scale: float = 0.5,
                 goal_threshold: float = 0.5,
                 render_mode: Optional[str] = None):
        """
        Initialize the Simple Dynamic Point Maze environment.
        
        Args:
            arena_size: Size of the square arena (goes from -arena_size/2 to +arena_size/2)
            max_episode_steps: Maximum steps per episode
            goal_reward: Reward for reaching the goal
            distance_reward_scale: Scale factor for distance-based reward
            action_scale: Scale factor for actions
            goal_threshold: Distance threshold for goal achievement
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.arena_size = arena_size
        self.max_episode_steps = max_episode_steps
        self.goal_reward = goal_reward
        self.distance_reward_scale = distance_reward_scale
        self.action_scale = action_scale
        self.goal_threshold = goal_threshold
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [agent_x, agent_y]
        arena_bound = arena_size / 2
        self.observation_space = spaces.Box(
            low=-arena_bound, high=arena_bound, shape=(2,), dtype=np.float64
        )
        
        # State variables
        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.goal_pos = np.zeros(2, dtype=np.float64)
        self.prev_distance = 0.0
        self.step_count = 0
        
        # For rendering
        self.fig = None
        self.ax = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Sample random agent position
        arena_bound = self.arena_size / 2
        self.agent_pos = self.np_random.uniform(
            low=-arena_bound * 0.8,  # Keep away from edges
            high=arena_bound * 0.8,
            size=2
        ).astype(np.float64)
        
        # Sample random goal position (ensure it's not too close to agent)
        min_goal_distance = 3.0
        max_attempts = 100
        
        for _ in range(max_attempts):
            self.goal_pos = self.np_random.uniform(
                low=-arena_bound * 0.8,
                high=arena_bound * 0.8,
                size=2
            ).astype(np.float64)
            
            distance = np.linalg.norm(self.goal_pos - self.agent_pos)
            if distance >= min_goal_distance:
                break
        
        # Initialize distance tracking
        self.prev_distance = np.linalg.norm(self.goal_pos - self.agent_pos)
        
        # Return observation and info
        obs = self.agent_pos.copy()
        info = {
            'goal': self.goal_pos.copy(),
            'distance_to_goal': self.prev_distance,
            'xy': self.agent_pos.copy()
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.step_count += 1
        
        # Apply action (scaled)
        action = np.array(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)  # Ensure action is in bounds
        
        # Update agent position
        self.agent_pos += action * self.action_scale
        
        # Keep agent within arena bounds
        arena_bound = self.arena_size / 2
        self.agent_pos = np.clip(self.agent_pos, -arena_bound, arena_bound)
        
        # Calculate distance to goal
        current_distance = np.linalg.norm(self.goal_pos - self.agent_pos)
        
        # Calculate reward
        reward = 0.0
        
        # Dense distance-based reward (reward for getting closer)
        distance_improvement = self.prev_distance - current_distance
        reward += distance_improvement * self.distance_reward_scale
        
        # Goal achievement reward
        goal_achieved = current_distance <= self.goal_threshold
        if goal_achieved:
            reward += self.goal_reward
        
        # Update previous distance
        self.prev_distance = current_distance
        
        # Check termination conditions
        terminated = goal_achieved
        truncated = self.step_count >= self.max_episode_steps
        
        # Create info dict
        info = {
            'success': float(goal_achieved),
            'distance_to_goal': current_distance,
            'xy': self.agent_pos.copy(),
            'qpos': self.agent_pos.copy(),
            'qvel': np.zeros(2),  # No velocity in this simple env
            'goal': self.goal_pos.copy()
        }
        
        return self.agent_pos.copy(), reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
            
        if self.render_mode == 'human':
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_xlim(-self.arena_size/2, self.arena_size/2)
                self.ax.set_ylim(-self.arena_size/2, self.arena_size/2)
                self.ax.set_aspect('equal')
                self.ax.grid(True, alpha=0.3)
                self.ax.set_title('Simple Dynamic Point Maze')
            
            self.ax.clear()
            self.ax.set_xlim(-self.arena_size/2, self.arena_size/2)
            self.ax.set_ylim(-self.arena_size/2, self.arena_size/2)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            
            # Draw agent
            self.ax.scatter(self.agent_pos[0], self.agent_pos[1], 
                          c='blue', s=100, marker='o', label='Agent')
            
            # Draw goal
            self.ax.scatter(self.goal_pos[0], self.goal_pos[1], 
                          c='red', s=150, marker='*', label='Goal')
            
            # Draw goal threshold circle
            circle = plt.Circle(self.goal_pos, self.goal_threshold, 
                              fill=False, color='red', linestyle='--', alpha=0.5)
            self.ax.add_patch(circle)
            
            # Add distance info
            distance = np.linalg.norm(self.goal_pos - self.agent_pos)
            self.ax.set_title(f'Step: {self.step_count}, Distance: {distance:.2f}')
            self.ax.legend()
            
            plt.pause(0.01)
            
        elif self.render_mode == 'rgb_array':
            # For rgb_array mode, create the plot and return as array
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(6, 6))
            
            self.ax.clear()
            self.ax.set_xlim(-self.arena_size/2, self.arena_size/2)
            self.ax.set_ylim(-self.arena_size/2, self.arena_size/2)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            
            # Draw agent and goal
            self.ax.scatter(self.agent_pos[0], self.agent_pos[1], 
                          c='blue', s=100, marker='o')
            self.ax.scatter(self.goal_pos[0], self.goal_pos[1], 
                          c='red', s=150, marker='*')
            
            # Convert to RGB array
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            return buf
    
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Register the environment
gym.register(
    id='SimpleDynamicPointMaze-v0',
    entry_point='simple_dynamic_pointmaze:SimpleDynamicPointMaze',
    max_episode_steps=500,
)


if __name__ == '__main__':
    # Test the environment
    env = SimpleDynamicPointMaze(render_mode=None)  # No visual rendering for test
    
    print("Testing Simple Dynamic Point Maze")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Goal position: {info['goal']}")
    print(f"Initial distance: {info['distance_to_goal']:.2f}")
    
    # Run a few random steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: reward={reward:.3f}, distance={info['distance_to_goal']:.2f}")
        
        if terminated:
            print("Goal reached!")
            break
        elif truncated:
            print("Episode truncated")
            break
    
    env.close()