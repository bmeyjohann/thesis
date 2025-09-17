#!/usr/bin/env python3
"""
Interactive Evaluation Script for RSL-RL Trained Agents

Load a trained RSL-RL policy and run it interactively on OGBench environments
with visual rendering. Perfect for testing trained agents and debugging!

Usage:
    # Plain policy evaluation
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-medium-v0

    # With human teleop intervention overlay
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-arena-danger-lethal-v0 \
        --intervention_mode human

    # With BFS teacher interventions
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-arena-danger-lethal-v0 \
        --intervention_mode agent --teacher_type bfs --tolerance_type angle --tolerance_value 30
    
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
from typing import Any, Dict

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
    parser.add_argument('--policy_type', type=str, default='auto',
                        choices=['auto', 'rsl-rl', 'fastsac'],
                        help='Policy checkpoint format to load')
    
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
    parser.add_argument('--print_interventions', action='store_true', default=True,
                        help='Print when the teacher intervenes and why')

    # Intervention / Teleop
    parser.add_argument('--intervention_mode', type=str, default='none',
                        choices=['none', 'human', 'agent'],
                        help='Intervention mode: none, human teleop, or agent teacher')
    parser.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs'],
                        help='Teacher type when intervention_mode=agent')
    parser.add_argument('--tolerance_type', type=str, default='angle', choices=['angle', 'l2'],
                        help='Intervention tolerance metric (agent mode)')
    parser.add_argument('--tolerance_value', type=float, default=30.0,
                        help='Tolerance threshold (deg for angle; abs for l2)')
    parser.add_argument('--hard_block_lethal', action='store_true', default=True,
                        help='Intervene if student would step into lethal cell')
    parser.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    
    # Action processing (should match training settings)
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Action scaling factor (should match training)')
    parser.add_argument('--clip_actions', action='store_true', default=True,
                        help='Clip actions to [-1, 1] (should match training)')
    
    return parser.parse_args()

class FastSACPolicy:
    """Thin wrapper that mimics ActorCritic.act() using the FastSAC actor."""

    def __init__(self, actor, obs_normalizer):
        self.actor = actor
        self.obs_normalizer = obs_normalizer
        self.device = next(actor.parameters()).device

    def eval(self):
        self.actor.eval()
        self.obs_normalizer.eval()

    def act(self, obs_dict, deterministic: bool = True):
        obs = obs_dict["policy"].to(self.device)
        with torch.no_grad():
            norm_obs = self.obs_normalizer(obs, center=True)
            actions, _, means = self.actor(norm_obs)
        return means if deterministic else actions


def _resolve_policy_type(args_policy_type: str, checkpoint: Dict[str, Any]) -> str:
    if args_policy_type != 'auto':
        return args_policy_type
    keys = set(checkpoint.keys())
    if {'actor_state_dict', 'qnet_state_dict'} <= keys:
        return 'fastsac'
    return 'rsl-rl'


def load_trained_policy(model_path: str, env, device: torch.device, args) -> Any:
    """Load a trained RSL-RL policy from checkpoint."""
    print(f"🔄 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint (trust local files)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print(f"✓ Checkpoint loaded")
    
    policy_type = _resolve_policy_type(args.policy_type, checkpoint)
    print(f"📦 Detected policy format: {policy_type}")

    training_info = {}

    if policy_type == 'rsl-rl':
        dummy_obs = torch.zeros(1, env.observation_space.shape[0], device=device)
        dummy_obs_dict = TensorDict({
            "policy": dummy_obs,
        }, batch_size=[1], device=device)

        if 'policy_cfg' in checkpoint:
            config = checkpoint['policy_cfg']
            print(f"📋 Model config found: {config}")
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"📋 Model config found: {config}")
        else:
            config = {
                'hidden_dims': [256, 256, 256],
                'activation': 'elu',
            }
            print(f"⚠️  No model config found, using defaults: {config}")

        valid_keys = {'hidden_dims', 'activation', 'init_noise_std', 'actor_hidden_dims', 'critic_hidden_dims'}
        config = {k: v for k, v in config.items() if k in valid_keys}

        try:
            policy = ActorCritic(
                obs=dummy_obs_dict,
                obs_groups={"policy": ["policy"], "critic": ["policy"]},
                num_actions=env.action_space.shape[0],
                **config,
            ).to(device)
            print(
                f"✓ ActorCritic created with obs shape: {dummy_obs.shape}, action dim: {env.action_space.shape[0]}"
            )
        except Exception as e:
            print(f"❌ ActorCritic creation failed: {e}")
            raise

        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"✓ Model weights loaded (policy_state_dict)")
        elif 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model weights loaded (model_state_dict)")
        elif 'state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['state_dict'])
            print(f"✓ Model weights loaded (state_dict)")
        else:
            try:
                policy.load_state_dict(checkpoint)
                print(f"✓ Model weights loaded (direct state dict)")
            except Exception as e:
                print(f"❌ Failed to load weights. Checkpoint keys: {list(checkpoint.keys())}")
                raise e

        policy.eval()
        print(f"✓ Policy set to evaluation mode")

    else:  # FastSAC checkpoint
        from fast_sac import Actor
        from fast_sac_utils import EmpiricalNormalization

        args_dict = checkpoint.get('args', {}) or {}
        if isinstance(args_dict, dict):
            actor_hidden = args_dict.get('actor_hidden_dim', 512)
            init_scale = args_dict.get('init_scale', 0.01)
        else:
            actor_hidden = getattr(args_dict, 'actor_hidden_dim', 512)
            init_scale = getattr(args_dict, 'init_scale', 0.01)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        actor = Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=1,
            init_scale=init_scale,
            hidden_dim=actor_hidden,
            device=device,
        ).to(device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        print(f"✓ FastSAC actor weights loaded")

        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_state = checkpoint.get('obs_normalizer_state')
        if obs_state:
            obs_normalizer.load_state_dict(obs_state)
        obs_normalizer.eval()

        policy = FastSACPolicy(actor, obs_normalizer)
        policy.eval()

    # Gather training metadata if available
    
    # Extract training info from different possible keys
    if 'training_info' in checkpoint:
        training_info.update(checkpoint['training_info'])
    
    # Additional info from checkpoint metadata
    for key in ['iteration', 'total_timesteps']:
        if key in checkpoint:
            training_info[key] = checkpoint[key]
    
    # Extract args if available
    if 'args' in checkpoint:
        args_obj = checkpoint['args']
        if isinstance(args_obj, dict):
            if 'env_name' in args_obj:
                training_info['env_name'] = args_obj['env_name']
            if 'reward_type' in args_obj:
                training_info['reward_type'] = args_obj['reward_type']
        else:
            if hasattr(args_obj, 'env_name'):
                training_info['env_name'] = args_obj.env_name
            if hasattr(args_obj, 'reward_type'):
                training_info['reward_type'] = args_obj.reward_type
    
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
        
        # Base observation wrapper
        from ogbench.wrappers import FlexibleObsWrapper, InterventionWrapper
        env = FlexibleObsWrapper(env, include_goal=True)  # Match training config
        print(f"✓ Applied FlexibleObsWrapper")

        # Optional intervention wrapper
        if args.intervention_mode == 'human':
            # Create control window teleop
            from ogbench.teleop import ControlWindowTeleop
            teleop = ControlWindowTeleop(width=520, height=420, show_debug_info=True)
            env = InterventionWrapper(
                env,
                teleop_interface=teleop,
                mode='human',
                threshold=0.1,
                hold_time=0.5,
            )
            print(f"✓ Applied InterventionWrapper (human teleop)")
        elif args.intervention_mode == 'agent':
            env = InterventionWrapper(
                env,
                mode='agent',
                teacher_type=args.teacher_type,
                tolerance_type=args.tolerance_type,
                tolerance_value=args.tolerance_value,
                hard_block_lethal=args.hard_block_lethal,
            )
            print(f"✓ Applied InterventionWrapper (agent teacher: {args.teacher_type})")
        
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
    step_idx = 0
    
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
        
        # Print intervention events
        step_idx += 1
        if args.print_interventions and isinstance(info, dict) and info.get('teacher_intervened', False):
            reason = info.get('teacher_reason', 'unknown')
            # compute angle between actions if available
            angle_str = ''
            try:
                sa = np.array(info.get('student_action'), dtype=np.float32)
                ta = np.array(info.get('teacher_action'), dtype=np.float32)
                if sa is not None and ta is not None:
                    an = np.linalg.norm(sa)
                    bn = np.linalg.norm(ta)
                    if an > 1e-8 and bn > 1e-8:
                        cos = float(np.clip(np.dot(sa, ta) / (an * bn), -1.0, 1.0))
                        angle = float(np.degrees(np.arccos(cos)))
                        angle_str = f", angle={angle:.1f} deg"
            except Exception:
                pass
            print(f"🛟 Teacher intervention at step {episode_length}: reason={reason}{angle_str}")

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
            
            # If teacher metrics available, print summary
            if isinstance(info, dict) and 'teacher_num_interventions' in info:
                print("   Teacher summary:")
                print(f"     interventions: {int(info['teacher_num_interventions'])}")
                print(f"     steps: {int(info['teacher_intervention_steps'])} / {int(info.get('teacher_episode_steps', episode_length))}")
                print(f"     fraction: {float(info['teacher_fraction_steps']):.3f}")
                print(f"     avg_burst_len: {float(info['teacher_avg_burst_len']):.2f}")
                print(f"     safety: {int(info['teacher_num_safety_interventions'])}, divergence: {int(info['teacher_num_divergence_interventions'])}")

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
        policy = load_trained_policy(args.model_path, env, device, args)
        
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
