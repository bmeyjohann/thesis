import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import pygame
import time

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center windows

# Import ogbench to register environments
import ogbench

# Import the intervention system
from ogbench.ui import TeleopPoint2D
from ogbench.wrappers import HumanInterventionWrapper

# Import our goal-conditioned wrapper and speed wrapper from ogbench
from ogbench.wrappers import RelativeGoalWrapper, SpeedWrapper

import sys
sys.path.append('fasttd3/fast_sac')

from fast_sac_utils import SimpleReplayBuffer, EmpiricalNormalization
from hyperparams import get_args
from fast_sac import Actor, Critic

class ControlWindow:
    """Persistent control window for human input and mode switching."""
    
    def __init__(self, width=400, height=300):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Training Control Panel")
        
        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        
        # Font
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.visual_mode_requested = False
        self.headless_mode_requested = False
        self.quit_requested = False
        
    def update(self, current_mode, episode, step, intervention_stats=None):
        """Update the control window display."""
        self.screen.fill(self.black)
        
        # Title
        title = self.font.render("FastSAC Training Control", True, self.white)
        self.screen.blit(title, (10, 10))
        
        # Current mode
        mode_color = self.green if current_mode == "visual" else self.blue
        mode_text = self.font.render(f"Mode: {current_mode.upper()}", True, mode_color)
        self.screen.blit(mode_text, (10, 40))
        
        # Episode/step info
        episode_text = self.small_font.render(f"Episode: {episode}, Step: {step}", True, self.white)
        self.screen.blit(episode_text, (10, 70))
        
        # Intervention stats
        if intervention_stats:
            intervention_pct = intervention_stats.get('percentage', 0)
            intervention_text = self.small_font.render(f"Human Intervention: {intervention_pct:.1f}%", True, self.white)
            self.screen.blit(intervention_text, (10, 90))
        
        # Controls
        controls_y = 120
        controls = [
            "CONTROLS:",
            "Arrow Keys: Move agent",
            "V: Switch to Visual mode",
            "H: Switch to Headless mode", 
            "ESC/Q: Quit training"
        ]
        
        for i, control in enumerate(controls):
            color = self.white if i == 0 else self.small_font.render("", True, self.white).get_rect().height
            text = self.small_font.render(control, True, self.white)
            self.screen.blit(text, (10, controls_y + i * 20))
        
        # Status messages
        status_y = 240
        if self.visual_mode_requested:
            status = self.small_font.render("Visual mode requested - will switch next episode", True, self.green)
            self.screen.blit(status, (10, status_y))
        elif self.headless_mode_requested:
            status = self.small_font.render("Headless mode requested - will switch next episode", True, self.blue)
            self.screen.blit(status, (10, status_y))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events and return any mode change requests."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self.visual_mode_requested = True
                    self.headless_mode_requested = False
                    print("Visual mode requested - will switch at next episode")
                elif event.key == pygame.K_h:
                    self.headless_mode_requested = True
                    self.visual_mode_requested = False
                    print("Headless mode requested - will switch at next episode")
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.quit_requested = True
        
        return {
            'visual_requested': self.visual_mode_requested,
            'headless_requested': self.headless_mode_requested,
            'quit_requested': self.quit_requested
        }
    
    def clear_requests(self):
        """Clear mode change requests after they've been processed."""
        self.visual_mode_requested = False
        self.headless_mode_requested = False
    
    def close(self):
        """Close the control window."""
        pygame.quit()

class TrainingLogger:
    """Clean logging system to track training progress without jumping output."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reset_stats()
        
    def reset_stats(self):
        self.rewards = []
        self.distance_rewards = []
        self.directional_rewards = []
        self.episode_lengths = []
        self.intervention_rates = []
        self.actions_x = []
        self.actions_y = []
        
    def add_step_data(self, reward, distance_reward, directional_reward, action):
        if len(self.distance_rewards) >= self.window_size:
            self.distance_rewards.pop(0)
            self.directional_rewards.pop(0)
            self.actions_x.pop(0)
            self.actions_y.pop(0)
            
        self.distance_rewards.append(distance_reward)
        self.directional_rewards.append(directional_reward)
        self.actions_x.append(action[0])
        self.actions_y.append(action[1])
        
    def add_episode_data(self, episode_reward, episode_length, intervention_rate):
        if len(self.rewards) >= self.window_size:
            self.rewards.pop(0)
            self.episode_lengths.pop(0)
            self.intervention_rates.pop(0)
            
        self.rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.intervention_rates.append(intervention_rate)
        
    def get_trends(self):
        if len(self.rewards) < 10:
            return "Insufficient data"
            
        recent_rewards = self.rewards[-20:] if len(self.rewards) >= 20 else self.rewards
        avg_reward = np.mean(recent_rewards)
        
        recent_dist_rewards = self.distance_rewards[-100:] if len(self.distance_rewards) >= 100 else self.distance_rewards
        avg_dist_reward = np.mean(recent_dist_rewards) if recent_dist_rewards else 0
        
        recent_dir_rewards = self.directional_rewards[-100:] if len(self.directional_rewards) >= 100 else self.directional_rewards
        avg_dir_reward = np.mean(recent_dir_rewards) if recent_dir_rewards else 0
        
        recent_actions_x = self.actions_x[-100:] if len(self.actions_x) >= 100 else self.actions_x
        recent_actions_y = self.actions_y[-100:] if len(self.actions_y) >= 100 else self.actions_y
        avg_action_x = np.mean(recent_actions_x) if recent_actions_x else 0
        avg_action_y = np.mean(recent_actions_y) if recent_actions_y else 0
        
        return {
            'avg_reward': avg_reward,
            'avg_dist_reward': avg_dist_reward,
            'avg_dir_reward': avg_dir_reward,
            'avg_action_x': avg_action_x,
            'avg_action_y': avg_action_y,
            'num_episodes': len(self.rewards)
        }

def create_environment(env_name, render_mode, max_episode_steps=500, goal_conditioned=True, speed_multiplier=3.0):
    """Create environment with specified render mode, goal conditioning, and speed."""
    try:
        if render_mode == "human":
            env = gym.make(
                env_name,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                width=800,
                height=600
            )
        else:
            env = gym.make(
                env_name,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps
            )
        
        # Apply speed wrapper first (consistent speed for uniform training)
        env = SpeedWrapper(env, speed_multiplier=speed_multiplier)
        
        # Apply goal-conditioned wrapper to fix observation space
        if goal_conditioned:
            env = RelativeGoalWrapper(env)
            print(f"✅ Applied RelativeGoalWrapper - agent can now see goal!")
        else:
            print(f"⚠️  No goal conditioning - agent cannot see goal position!")
            
        return env
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return None

def calculate_distance_reward(obs, prev_distance=None, enable_reward_shaping=False):
    """
    Calculates distance-based reward from a goal-conditioned observation.
    Mirrored from the working debug script to remove confusing signals.
    """
    if not enable_reward_shaping or not hasattr(obs, '__len__') or len(obs) != 4:
        return 0.0, None

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

def calculate_directional_reward(obs, action, enable_reward_shaping=False):
    """
    Calculates reward based on how well the action aligns with the goal direction.
    Mirrored from the working debug script.
    """
    if not enable_reward_shaping or not hasattr(obs, '__len__') or len(obs) != 4 or np.linalg.norm(action) < 0.01:
        return 0.0

    relative_goal = obs[2:4]
    goal_distance = np.linalg.norm(relative_goal)
    if goal_distance < 0.01:
        return 0.5 # At goal bonus

    desired_direction = relative_goal / goal_distance
    
    # Normalize action to get actual direction
    action_magnitude = np.linalg.norm(action)
    if action_magnitude < 0.01: # No movement
        return 0.0
    action_direction = action / action_magnitude
    
    alignment = np.dot(desired_direction, action_direction)
    
    # Convert alignment to reward (stronger signal)
    return alignment * 2.0

def debug_observation_space(env, obs, info, step_count=0, action=None):
    """Debug function to understand what's in the observation space."""
    if step_count % 50 == 0:  # Print every 50 steps to avoid spam
        print(f"\n=== OBSERVATION DEBUG (Step {step_count}) ===")
        print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        print(f"Observation value: {obs}")
        print(f"Info keys: {list(info.keys()) if isinstance(info, dict) else 'No info dict'}")
        
        # Enhanced debugging for goal-conditioned observations
        if hasattr(obs, '__len__') and len(obs) == 4:
            agent_pos = obs[:2]
            relative_goal = obs[2:4]
            distance = np.linalg.norm(relative_goal)
            
            print(f"Agent position: [{agent_pos[0]:.2f}, {agent_pos[1]:.2f}]")
            print(f"Relative goal: [{relative_goal[0]:.2f}, {relative_goal[1]:.2f}]")
            print(f"Distance to goal: {distance:.3f}")
            
            # Show optimal action direction
            optimal_action_x = "LEFT" if relative_goal[0] < 0 else "RIGHT"
            optimal_action_y = "DOWN" if relative_goal[1] < 0 else "UP"
            print(f"Agent should move: {optimal_action_x} and {optimal_action_y}")
            
            # Show action info (from wrapper or parameter)
            action_to_analyze = None
            if 'policy_action_used' in info:
                action_to_analyze = info['policy_action_used']
            elif action is not None:
                action_to_analyze = action
                
            if action_to_analyze is not None:
                actual_x = "RIGHT" if action_to_analyze[0] > 0 else "LEFT"
                actual_y = "UP" if action_to_analyze[1] > 0 else "DOWN"
                print(f"Actual action: [{action_to_analyze[0]:.3f}, {action_to_analyze[1]:.3f}] ({actual_x}, {actual_y})")
                
                # Check if action is in correct direction
                correct_x = (relative_goal[0] < 0 and action_to_analyze[0] < 0) or (relative_goal[0] > 0 and action_to_analyze[0] > 0)
                correct_y = (relative_goal[1] < 0 and action_to_analyze[1] < 0) or (relative_goal[1] > 0 and action_to_analyze[1] > 0)
                
                if correct_x and correct_y:
                    print("✅ Action moving toward goal")
                elif correct_x or correct_y:
                    print("⚠️  Action partially toward goal")
                else:
                    print("❌ Action moving AWAY from goal!")
            else:
                print("No action info available for analysis")
        
        # Print other useful info
        if isinstance(info, dict):
            for key in ['xy', 'prev_qpos', 'prev_qvel', 'qpos', 'qvel', 'success', 'human_override', 'intervene_action', 'policy_action_ignored', 'policy_action_used']:
                if key in info:
                    print(f"Info[{key}]: {info[key]}")
        print("=" * 50)

def main():
    # Handle custom arguments before tyro sees them
    import sys
    
    # Extract our custom arguments
    initial_mode = 'visual'
    total_timesteps = 100_000
    num_envs = 1
    enable_reward_shaping = False  # Add reward shaping toggle
    enable_goal_conditioning = True  # Enable goal conditioning by default
    
    # Parse and remove custom arguments from sys.argv
    args_to_remove = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--initial-mode' and i + 1 < len(sys.argv):
            initial_mode = sys.argv[i + 1]
            args_to_remove.extend([i, i + 1])
            i += 2
        elif sys.argv[i] == '--total-timesteps' and i + 1 < len(sys.argv):
            total_timesteps = int(sys.argv[i + 1])
            args_to_remove.extend([i, i + 1])
            i += 2
        elif sys.argv[i] == '--num-envs' and i + 1 < len(sys.argv):
            num_envs = int(sys.argv[i + 1])
            args_to_remove.extend([i, i + 1])
            i += 2
        elif sys.argv[i] == '--enable-reward-shaping':
            enable_reward_shaping = True
            args_to_remove.append(i)
            i += 1
        elif sys.argv[i] == '--disable-goal-conditioning':
            enable_goal_conditioning = False
            args_to_remove.append(i)
            i += 1
        else:
            i += 1
    
    # Remove custom arguments from sys.argv (in reverse order to maintain indices)
    for idx in sorted(args_to_remove, reverse=True):
        sys.argv.pop(idx)
    
    # Show help for our custom arguments if requested
    if '--help' in sys.argv:
        print("FastSAC PointMaze Training with Dynamic Mode Switching")
        print("\nCustom Arguments:")
        print("  --initial-mode {visual,headless}  Initial rendering mode (default: visual)")
        print("  --total-timesteps INT            Total training timesteps (default: 100000)")
        print("  --num-envs INT                   Number of parallel environments (default: 1)")
        print("  --enable-reward-shaping          Enable distance-based reward shaping (default: disabled)")
        print("  --disable-goal-conditioning      Disable goal-conditioned observations (default: enabled)")
        print("\nDuring training:")
        print("  V key: Switch to visual mode")
        print("  H key: Switch to headless mode")
        print("  Arrow keys: Control agent (visual mode only)")
        print("  ESC/Q: Exit")
        return

            # Get FastSAC args and override environment settings
    args = get_args()
    args.env_name = "pointmaze-medium-v0"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.total_timesteps = total_timesteps
    args.learning_starts = 1_000
    args.batch_size = 32768  # Fixed: was 256, now matches reference
    args.buffer_size = 1_000_000
    args.actor_learning_rate = 3e-4  # Fixed: was 1e-3, now matches reference
    args.critic_learning_rate = 3e-4  # Fixed: was 1e-3, now matches reference
    args.gamma = 0.99
    args.tau = 0.1  # Fixed: was 0.005, now matches reference (20x increase)
    # SAC doesn't use policy noise
    args.noise_clip = 0.5
    args.policy_frequency = 2
    args.seed = 0

    print(f"Using device: {args.device}")
    print(f"🔧 Using reference fast_sac hyperparameters:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rates: {args.actor_learning_rate}")
    print(f"   Tau: {args.tau}")

    # Create control window
    control_window = ControlWindow()
    
    # Initialize teleoperation interface
    teleop_agent = TeleopPoint2D(deadzone=0.15, use_keyboard_fallback=True)
    teleop_agent.print_controls()
    
    # Create initial environment
    current_mode = initial_mode
    render_mode = "human" if current_mode == "visual" else None
    env = create_environment(args.env_name, render_mode, goal_conditioned=enable_goal_conditioning)
    
    if env is None:
        control_window.close()
        return

    # Apply intervention wrapper only in visual mode
    if current_mode == "visual":
        env = HumanInterventionWrapper(
            env, 
            teleop_agent,
            threshold=0.2,
            hold_time=0.5
        )
        print("Human intervention enabled (visual mode)")
    else:
        print("Running in headless mode - no intervention")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Environment: {args.env_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    
    print(f"Observation dimensions: {obs_dim}")
    print(f"Action dimensions: {act_dim}")

    # Initialize observation normalizer
    obs_normalizer = nn.Identity()
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=args.device)
        print("✅ Observation normalization enabled.")

    def process_obs(obs):
        """Convert observation to tensor"""
        return torch.tensor(obs, dtype=torch.float32, device=args.device)

    # Initialize networks
    actor = Actor(obs_dim, act_dim, num_envs=num_envs, device=args.device, init_scale=1.0, hidden_dim=512).to(args.device)  # Fixed: was 256, now 512

    critic = Critic(obs_dim, act_dim, hidden_dim=1024, device=args.device).to(args.device)  # Fixed: was 256, now 1024
    critic_target = Critic(obs_dim, act_dim, hidden_dim=1024, device=args.device).to(args.device)  # Fixed: was 256, now 1024
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    
    # SAC: Alpha parameter for entropy regularization
    target_entropy = -float(act_dim)
    log_alpha = torch.ones(1, requires_grad=True, device=args.device)
    log_alpha.data.copy_(torch.tensor([np.log(0.001)], device=args.device))
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.critic_learning_rate)

    replay = SimpleReplayBuffer(
        n_env=num_envs,
        buffer_size=args.buffer_size,
        n_obs=obs_dim,
        n_act=act_dim,
        n_critic_obs=obs_dim,
        asymmetric_obs=False,
        device=args.device,
    )

    # Training state
    obs, info = env.reset(seed=args.seed)
    obs = process_obs(obs)
    episode_reward = 0
    global_step = 0
    episode_count = 0
    
    # Intervention tracking
    episode_intervention_steps = 0
    episode_total_steps = 0
    
    # Initialize distance tracking for first episode
    if enable_reward_shaping:
        obs_array = obs.cpu().numpy() if hasattr(obs, 'cpu') else obs
        if len(obs_array) == 4:
            prev_distance = np.linalg.norm(obs_array[2:4])
        else:
            prev_distance = None

    print(f"\n=== DYNAMIC TRAINING MODE ===")
    print(f"🤖 Training FastSAC on OGBench PointMaze")
    print(f"📺 Starting in {current_mode} mode")
    print(f"🎮 Control window provides intervention and mode switching")
    print(f"🎯 Goal conditioning: {'ENABLED' if enable_goal_conditioning else 'DISABLED'}")
    print(f"💰 Reward shaping: {'ENABLED' if enable_reward_shaping else 'DISABLED'}")
    print(f"🔄 V = Visual mode, H = Headless mode, ESC = Quit")

    logger = TrainingLogger()
    
    # Track distance for improvement-based rewards
    prev_distance = None

    while global_step < args.total_timesteps:
        # Handle control window events
        events = control_window.handle_events()
        
        if events['quit_requested']:
            print("\nQuitting training...")
            break
        
        # Check for mode switching at episode boundaries
        mode_switch_requested = events['visual_requested'] or events['headless_requested']
        
        # Get action
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                norm_obs = obs_normalizer(obs.unsqueeze(0)).squeeze(0)
                action, _, _ = actor(norm_obs.unsqueeze(0))
                action = action.cpu().numpy()[0]

        # Store numpy observation for reward calculation (before tensor conversion)
        obs_numpy = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        
        # Step environment
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        # Debug observation space occasionally (reduced frequency for cleaner output)
        if global_step % 200 == 0:  # Much less frequent debugging
            debug_observation_space(env, next_obs, step_info, global_step, action)
        
        # Add shaped rewards if enabled (using the corrected reward logic)
        obs_numpy_for_reward = next_obs if isinstance(next_obs, np.ndarray) else next_obs.cpu().numpy()
        action_numpy = action if isinstance(action, np.ndarray) else action.cpu().numpy()

        distance_reward, current_distance = calculate_distance_reward(
            obs_numpy_for_reward, prev_distance, enable_reward_shaping
        )
        directional_reward = calculate_directional_reward(
            obs_numpy, action_numpy, enable_reward_shaping
        )
        total_reward = reward + distance_reward + directional_reward
        
        # Update previous distance for next step
        if current_distance is not None:
            prev_distance = current_distance
        
        # Log reward components occasionally
        if global_step % 100 == 0 and enable_reward_shaping:
            print(f"Rewards - Sparse: {reward:.3f}, Distance: {distance_reward:.3f}, Directional: {directional_reward:.3f}, Total: {total_reward:.3f}")
        
        next_obs = process_obs(next_obs)

        # Track intervention stats
        episode_total_steps += 1
        if step_info.get("human_override", False):
            episode_intervention_steps += 1

        # Store transition
        transition = {
            "observations": obs,
            "actions": torch.tensor(action, dtype=torch.float32, device=args.device),
            "next": {
                "observations": next_obs,
                "rewards": torch.tensor([total_reward], dtype=torch.float32, device=args.device),
                "truncations": torch.tensor([truncated], dtype=torch.int, device=args.device),
                "dones": torch.tensor([terminated], dtype=torch.int, device=args.device),
            },
        }
        replay.extend(transition)

        obs = next_obs
        global_step += 1
        episode_reward += total_reward

        # Update control window
        intervention_stats = {
            'percentage': (episode_intervention_steps / episode_total_steps * 100) if episode_total_steps > 0 else 0
        }
        control_window.update(current_mode, episode_count, global_step, intervention_stats)

        # Log training data
        logger.add_step_data(total_reward, distance_reward, directional_reward, action)
        logger.add_episode_data(episode_reward, episode_total_steps, intervention_stats['percentage'])

        # Handle episode termination and mode switching
        if done:
            intervention_pct = (episode_intervention_steps / episode_total_steps * 100) if episode_total_steps > 0 else 0
            
            # Add episode data to logger
            logger.add_episode_data(episode_reward, episode_total_steps, intervention_pct)
            
            # Show episode summary with trends every 10 episodes
            if episode_count % 10 == 0:
                trends = logger.get_trends()
                if isinstance(trends, dict):
                    print(f"\n📊 TRAINING PROGRESS (Episodes {episode_count-9}-{episode_count}):")
                    print(f"   Average Reward: {trends['avg_reward']:.2f}")
                    print(f"   Distance Reward: {trends['avg_dist_reward']:.3f}")
                    print(f"   Directional Reward: {trends['avg_dir_reward']:.3f}")
                    print(f"   Action Bias: X={trends['avg_action_x']:.2f}, Y={trends['avg_action_y']:.2f}")
                    
                    # Diagnosis
                    if abs(trends['avg_action_x']) > 0.7 or abs(trends['avg_action_y']) > 0.7:
                        print(f"   ⚠️  BIAS DETECTED: Agent favoring corner movement!")
                    if trends['avg_dir_reward'] < -0.2:
                        print(f"   ❌ WRONG DIRECTION: Agent consistently moving away from goals!")
                    if trends['avg_reward'] > trends['avg_reward'] * 0.8:  # Simple improvement check
                        print(f"   ✅ IMPROVING: Rewards trending upward")
                else:
                    print(f"\nEpisode {episode_count}: Steps={episode_total_steps}, Reward={episode_reward:.2f}, "
                          f"Intervention={intervention_pct:.1f}%")
            else:
                # Brief episode summary
                print(f"Ep {episode_count}: R={episode_reward:.1f}, Steps={episode_total_steps}, Int={intervention_pct:.0f}%")
            
            # Handle mode switching at episode boundary
            if mode_switch_requested:
                try:
                    env.close()
                except:
                    pass
                
                # Switch mode
                if events['visual_requested']:
                    current_mode = "visual"
                    render_mode = "human"
                    print("Switching to visual mode...")
                elif events['headless_requested']:
                    current_mode = "headless"
                    render_mode = None
                    print("Switching to headless mode...")
                
                # Create new environment
                env = create_environment(args.env_name, render_mode, goal_conditioned=enable_goal_conditioning)
                if env is None:
                    break
                
                # Apply intervention wrapper only in visual mode
                if current_mode == "visual":
                    env = HumanInterventionWrapper(
                        env, 
                        teleop_agent,
                        threshold=0.2,
                        hold_time=0.5
                    )
                    print("Human intervention enabled")
                else:
                    print("Human intervention disabled (headless mode)")
                
                control_window.clear_requests()
            
            # Reset episode
            obs, info = env.reset()
            obs = process_obs(obs)
            episode_reward = 0
            episode_count += 1
            episode_intervention_steps = 0
            episode_total_steps = 0
            prev_distance = None  # Reset distance tracking for new episode

        # Training updates
        if global_step >= args.learning_starts:
            batch = replay.sample(args.batch_size)

            # Normalize observations from the replay buffer before they are used
            with torch.no_grad():
                obs_b = obs_normalizer(batch["observations"])
                next_obs_b = obs_normalizer(batch["next"]["observations"])

            act_b = batch["actions"]
            rew_b = batch["next"]["rewards"]
            done_b = batch["next"]["dones"]

            with torch.no_grad():
                # SAC: Get next actions and log probs from actor
                next_action, next_log_prob, _ = actor(next_obs_b)
                target_q1, target_q2 = critic_target(next_obs_b, next_action)
                target_q = torch.min(target_q1, target_q2)
                # SAC: Subtract log prob for entropy regularization
                target = rew_b + args.gamma * (1 - done_b) * (target_q - log_alpha.exp() * next_log_prob)
            
            # The critic sees the normalized current observations
            q1, q2 = critic(obs_b, act_b)
            critic_loss = ((q1 - target).pow(2).mean() + (q2 - target).pow(2).mean())

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if global_step % args.policy_frequency == 0:
                # SAC: Actor update
                action, log_prob, _ = actor(obs_b)
                q1, q2 = critic(obs_b, action)
                q_value = torch.min(q1, q2)
                actor_loss = (log_alpha.exp() * log_prob - q_value).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # SAC: Alpha update
                alpha_loss = -log_alpha.exp() * (log_prob + target_entropy).detach().mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # SAC: Only update critic target (no actor target in SAC)
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # Small delay to prevent overwhelming the system
        time.sleep(0.01)

    print("Training completed!")
    
    # Cleanup
    try:
        env.close()
    except:
        pass
    control_window.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        pygame.quit() 