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
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pygame
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'


class PixelNormalizer(nn.Module):
    def forward(self, x):
        return x / 255.0


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class PixelBackbone(nn.Module):
    def __init__(self, input_shape, feature_dim: int):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(),
        )
        self.output_dim = feature_dim

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class GaussianPolicyHead(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, init_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim // 2, action_dim)
        nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)

    def forward(self, features):
        x = self.net(features)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

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
    
    # Observation configuration (match training)
    parser.add_argument('--include_goal', dest='include_goal', action='store_true', default=True,
                        help='Include goal coordinates in observations')
    parser.add_argument('--no_include_goal', dest='include_goal', action='store_false',
                        help='Exclude goal coordinates from observations')
    parser.add_argument('--include_distance', action='store_true', default=False,
                        help='Include distance to goal in observations')
    parser.add_argument('--include_direction', action='store_true', default=False,
                        help='Include direction to goal in observations')
    parser.add_argument('--include_velocity', action='store_true', default=False,
                        help='Include velocity features in observations')

    # Reward shaping (should mirror training wrapper settings)
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['sparse', 'dense', 'combined'],
                        help='Reward type for DetailedRewardWrapper')
    parser.add_argument('--dense_reward_scale', type=float, default=0.01,
                        help='Scale for dense reward shaping (if applicable)')
    parser.add_argument('--step_penalty', type=float, default=0.0,
                        help='Per-step penalty applied by DetailedRewardWrapper')
    parser.add_argument('--reward_switch_after_steps', type=int, default=0,
                        help='Switch reward to sparse after this many steps (curriculum)')

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
    parser.add_argument('--intervention_enable_after_steps', type=int, default=0,
                        help='Warm-up steps before agent teacher interventions engage')
    
    # Action processing (should match training settings)
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Action scaling factor (should match training)')
    parser.add_argument('--clip_actions', action='store_true', default=True,
                        help='Clip actions to [-1, 1] (should match training)')
    
    return parser.parse_args()

class FastSACPolicy:
    """Unified wrapper for FastSAC policies (legacy and new architectures)."""

    def __init__(self, *,
                 obs_normalizer: nn.Module,
                 obs_mode: str,
                 pixel_shape=None,
                 actor_backbone: nn.Module | None = None,
                 actor_head: nn.Module | None = None,
                 legacy_actor: nn.Module | None = None):
        self.obs_normalizer = obs_normalizer
        self.obs_mode = obs_mode
        self.pixel_shape = pixel_shape
        self.actor_backbone = actor_backbone
        self.actor_head = actor_head
        self.legacy_actor = legacy_actor
        module = legacy_actor if legacy_actor is not None else actor_head
        self.device = next(module.parameters()).device

    def eval(self):
        self.obs_normalizer.eval()
        if self.legacy_actor is not None:
            self.legacy_actor.eval()
        else:
            self.actor_backbone.eval()
            self.actor_head.eval()

    def _normalize(self, obs):
        try:
            return self.obs_normalizer(obs, center=True)
        except TypeError:
            return self.obs_normalizer(obs)

    def act(self, obs_dict, deterministic: bool = True):
        obs = obs_dict["policy"].to(self.device)
        if self.legacy_actor is not None:
            with torch.no_grad():
                norm_obs = self._normalize(obs)
                actions, _, means = self.legacy_actor(norm_obs)
            return means if deterministic else actions

        norm_obs = self._normalize(obs)
        if self.obs_mode == "pixels":
            assert self.pixel_shape is not None, "pixel_shape must be provided for pixel observations"
            obs_input = norm_obs.view(norm_obs.shape[0], *self.pixel_shape)
        else:
            obs_input = norm_obs
        with torch.no_grad():
            features = self.actor_backbone(obs_input)
            actions, _, means = self.actor_head(features)
        return means if deterministic else actions


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
    if {'actor_backbone', 'actor_head'} <= keys:
        return 'fastsac_v2'
    if {'actor_state_dict', 'qnet_state_dict'} <= keys:
        return 'fastsac'
    return 'rsl-rl'


def load_trained_policy(model_path: str, env, device: torch.device, args):
    """Load a trained policy and return (policy, training_metadata)."""
    print(f"\n🔄 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("✓ Checkpoint loaded")

    policy_type = _resolve_policy_type(args.policy_type, checkpoint)
    print(f"📦 Detected policy format: {policy_type}")

    training_info: Dict[str, Any] = {}
    obs_space = env.observation_space
    act_dim = env.action_space.shape[0]
    obs_dim = int(np.prod(obs_space.shape))

    if policy_type == 'rsl-rl':
        dummy_obs = torch.zeros(1, obs_dim, device=device)
        dummy_obs_dict = TensorDict({"policy": dummy_obs}, batch_size=[1], device=device)
        config = checkpoint.get('policy_cfg') or checkpoint.get('model_config') or {
            'hidden_dims': [256, 256, 256],
            'activation': 'elu',
        }
        valid_keys = {'hidden_dims', 'activation', 'init_noise_std', 'actor_hidden_dims', 'critic_hidden_dims'}
        config = {k: v for k, v in config.items() if k in valid_keys}
        policy = ActorCritic(
            obs=dummy_obs_dict,
            obs_groups={"policy": ["policy"], "critic": ["policy"]},
            num_actions=act_dim,
            **config,
        ).to(device)
        state_dict = checkpoint.get('policy_state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        policy.load_state_dict(state_dict)
        policy.eval()
        print("✓ ActorCritic policy loaded")
    elif policy_type == 'fastsac':
        actor_hidden = checkpoint.get('args', {}).get('actor_hidden_dim', 512)
        init_scale = checkpoint.get('args', {}).get('init_scale', 0.01)
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
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        if checkpoint.get('obs_normalizer_state'):
            obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])
        obs_normalizer.eval()
        policy = FastSACPolicy(
            obs_normalizer=obs_normalizer,
            obs_mode='state',
            pixel_shape=None,
            legacy_actor=actor,
        )
        policy.eval()
    else:  # fastsac_v2
        train_args = checkpoint.get('args', {}) or {}
        obs_mode = train_args.get('obs_mode', getattr(args, 'obs_mode', 'state'))
        arch_shared = train_args.get('arch_shared_trunk', False)
        actor_hidden = train_args.get('actor_hidden_dim', 512)
        shared_hidden = train_args.get('shared_hidden_dim', actor_hidden)
        init_scale = train_args.get('init_scale', 0.01)
        feature_dim = shared_hidden if arch_shared else actor_hidden
        if obs_mode == 'pixels':
            pixel_shape = checkpoint.get('pixel_shape') or obs_space.shape
            backbone = PixelBackbone((pixel_shape[2], pixel_shape[0], pixel_shape[1]), feature_dim).to(device)
            obs_normalizer = PixelNormalizer().to(device)
        else:
            pixel_shape = None
            backbone = MLPBackbone(obs_dim, feature_dim).to(device)
            obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
            if checkpoint.get('obs_normalizer_state'):
                obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])
        backbone.load_state_dict(checkpoint['actor_backbone'])
        backbone.eval()
        actor_head = GaussianPolicyHead(backbone.output_dim, act_dim, actor_hidden, init_scale).to(device)
        actor_head.load_state_dict(checkpoint['actor_head'])
        actor_head.eval()
        policy = FastSACPolicy(
            obs_normalizer=obs_normalizer,
            obs_mode=obs_mode,
            pixel_shape=pixel_shape,
            actor_backbone=backbone,
            actor_head=actor_head,
        )
        policy.eval()
        training_info['obs_mode'] = obs_mode
        training_info['num_critics'] = train_args.get('num_critics', 2)
        training_info['arch_shared_trunk'] = arch_shared

    # Common training metadata
    args_obj = checkpoint.get('args')
    if isinstance(args_obj, dict):
        for key in ['env_name', 'reward_type', 'obs_mode', 'shared_hidden_dim']:
            if key in args_obj:
                training_info[key] = args_obj[key]
    elif args_obj is not None:
        for key in ['env_name', 'reward_type', 'obs_mode', 'shared_hidden_dim']:
            if hasattr(args_obj, key):
                training_info[key] = getattr(args_obj, key)
    for key in ['training_info', 'iteration', 'total_timesteps']:
        if key in checkpoint:
            training_info[key] = checkpoint[key]

    print("✓ Policy set to evaluation mode")
    return policy, training_info


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
        from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, InterventionWrapper
        if getattr(args, 'obs_mode', 'state') == 'state':
            env = FlexibleObsWrapper(
                env,
                include_goal=args.include_goal,
                include_distance=args.include_distance,
                include_direction=args.include_direction,
                include_velocity=args.include_velocity,
            )
            print('Applied FlexibleObsWrapper')
        env = DetailedRewardWrapper(
            env,
            reward_type=args.reward_type,
            dense_reward_scale=args.dense_reward_scale,
            step_penalty=args.step_penalty,
            switch_reward_to_sparse_after_steps_per_env=args.reward_switch_after_steps,
        )
        print('Applied DetailedRewardWrapper (type={})'.format(args.reward_type))

        if args.intervention_mode == 'human':
            from ogbench.teleop import ControlWindowTeleop
            teleop = ControlWindowTeleop(width=520, height=420, show_debug_info=True)
            env = InterventionWrapper(
                env,
                teleop_interface=teleop,
                mode='human',
                threshold=0.1,
                hold_time=0.5,
            )
            print('Applied InterventionWrapper (human teleop)')
        elif args.intervention_mode == 'agent':
            env = InterventionWrapper(
                env,
                mode='agent',
                teacher_type=args.teacher_type,
                tolerance_type=args.tolerance_type,
                tolerance_value=args.tolerance_value,
                hard_block_lethal=args.hard_block_lethal,
                enable_after_steps=args.intervention_enable_after_steps,
            )
            print('Applied InterventionWrapper (agent teacher: {})'.format(args.teacher_type))

        print('Environment created successfully')
        print('   Observation space:', env.observation_space)
        print('   Action space:', env.action_space)
        return env
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
        # Peek at checkpoint to recover observation mode
        checkpoint_preview = torch.load(args.model_path, map_location='cpu')
        args.obs_mode = checkpoint_preview.get('args', {}).get('obs_mode', getattr(args, 'obs_mode', 'state'))
        del checkpoint_preview

        # Create environment
        env = create_env(args.env_name, args)

        # Load trained policy
        policy, training_info = load_trained_policy(args.model_path, env, device, args)
        if training_info:
            print('Training info:')
            for key, value in training_info.items():
                print('   {}: {}'.format(key, value))

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
