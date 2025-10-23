#!/usr/bin/env python3
"""
RSL-RL Integrated Training Script for OGBench environments.

Uses proper RSL-RL OnPolicyRunner with config-driven approach.
Features:
- Full RSL-RL integration with OnPolicyRunner
- Config-driven training parameters
- Proper wandb integration with offline support
- Support for all OGBench environments
- Comprehensive metrics logging
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add RSL-RL to path
sys.path.append('rsl_rl')

# Set WANDB environment variables before any imports
os.environ.setdefault('WANDB_MODE', 'offline')
os.environ.setdefault('WANDB_CONSOLE', 'off')
os.environ.setdefault('WANDB_SILENT', 'true')

import torch
import numpy as np
import gymnasium as gym

# Import RSL-RL components
from rsl_rl.runners import OnPolicyRunner

# Import our OGBench components
import ogbench
from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, VectorizedOGBenchEnv, InterventionWrapper


def get_available_environments():
    """Get list of available OGBench environments."""
    return [
        'pointmaze-arena-v0',
        'pointmaze-arena-danger-floor-v0',
        'pointmaze-arena-danger-sticky-v0',
        'pointmaze-arena-danger-wall-v0',
        'pointmaze-arena-danger-lethal-v0',
        'pointmaze-medium-v0', 
        'pointmaze-large-v0',
        'pointmaze-giant-v0',
        'antmaze-medium-v0',
        'antmaze-large-v0',
        'humanoidmaze-medium-v0',
        'humanoidmaze-large-v0',
    ]


def create_env(env_name: str, num_envs: int, include_goal: bool, include_distance: bool,
               include_direction: bool, include_velocity: bool, reward_type: str,
               dense_reward_scale: float, step_penalty: float,
               # subgoal shaping + curriculum
               use_subgoal_shaping: bool = False,
               subgoal_shaping_coef: float = 1.0,
               subgoal_shaping_gamma: float = 0.99,
               curriculum_stage1_steps: int = 0,
               reward_switch_after_steps: int = 0,
               clip_actions: float = None,
               render_mode: str = None, max_episode_steps: int = None,
               # intervention wrapper
               use_intervention: bool = False,
               intervention_mode: str = 'agent',
               teacher_type: str = 'bfs',
               tolerance_type: str = 'angle',
               tolerance_value: float = 30.0,
               hard_block_lethal: bool = True,
               intervention_enable_after_steps: int = 0,
               ):
    """Create a vectorized OGBench environment with proper wrappers."""
    
    def apply_wrappers(env):
        """Apply wrappers to a single environment."""
        # Apply flexible observation wrapper
        env = FlexibleObsWrapper(
            env,
            include_goal=include_goal,
            include_distance=include_distance,
            include_direction=include_direction,
            include_velocity=include_velocity,
        )
        
        # Apply detailed reward wrapper
        env = DetailedRewardWrapper(
            env,
            reward_type=reward_type,
            dense_reward_scale=dense_reward_scale,
            step_penalty=step_penalty,
            use_subgoal_shaping=use_subgoal_shaping,
            subgoal_shaping_coef=subgoal_shaping_coef,
            subgoal_shaping_gamma=subgoal_shaping_gamma,
            curriculum_stage1_steps_per_env=(curriculum_stage1_steps // max(1, num_envs)),
            log_subgoal_metrics=True,
            switch_reward_to_sparse_after_steps_per_env=(reward_switch_after_steps // max(1, num_envs)),
        )
        
        # Apply intervention wrapper (teacher-student) if enabled
        if use_intervention:
            if intervention_mode == 'human':
                # Training usually won't use human teleop; skip with a warning
                print("[create_env] Human intervention requested during training—skipping (no teleop).")
            else:
                env = InterventionWrapper(
                    env,
                    mode='agent',
                    teacher_type=teacher_type,
                    tolerance_type=tolerance_type,
                    tolerance_value=tolerance_value,
                    hard_block_lethal=hard_block_lethal,
                    enable_after_steps=intervention_enable_after_steps,
                )
        
        return env
    
    # Environment kwargs
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs['render_mode'] = render_mode
    if max_episode_steps is not None:
        env_kwargs['max_episode_steps'] = max_episode_steps
    
    # Create vectorized environment
    env = VectorizedOGBenchEnv(
        env_name=env_name,
        num_envs=num_envs,
        wrappers=[apply_wrappers],
        clip_actions=clip_actions,
        **env_kwargs
    )
    
    return env


def load_config(config_path: str) -> dict:
    """Load RSL-RL configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RSL-RL Integrated Training for OGBench')
    
    # Environment
    parser.add_argument('--env_name', type=str, default='pointmaze-arena-v0',
                        choices=get_available_environments(),
                        help='Environment name')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel environments')
    parser.add_argument('--max_episode_steps', type=int, default=None,
                        help='Maximum episode steps')
    
    # Observations
    parser.add_argument('--include_goal', action='store_true', default=True,
                        help='Include goal position in observations')
    parser.add_argument('--include_distance', action='store_true', default=False,
                        help='Include distance to goal')
    parser.add_argument('--include_direction', action='store_true', default=False,
                        help='Include direction to goal')
    parser.add_argument('--include_velocity', action='store_true', default=False,
                        help='Include velocity information')
    
    # Rewards
    parser.add_argument('--reward_type', type=str, default='sparse',
                        choices=['sparse', 'dense', 'combined'],
                        help='Reward type')
    parser.add_argument('--dense_reward_scale', type=float, default=0.01,
                        help='Scale factor for dense rewards')
    parser.add_argument('--step_penalty', type=float, default=0.0,
                        help='Small penalty per step')
    # Subgoal shaping + curriculum (reward-only)
    parser.add_argument('--use_subgoal_shaping', action='store_true', default=False,
                        help='Enable potential-based shaping towards oracle subgoals (reward-only)')
    parser.add_argument('--subgoal_shaping_coef', type=float, default=1.0,
                        help='Coefficient for subgoal shaping term')
    parser.add_argument('--subgoal_shaping_gamma', type=float, default=0.99,
                        help='Gamma used in potential-based shaping term')
    parser.add_argument('--curriculum_stage1_steps', type=int, default=0,
                        help='Hard cutoff steps for Stage 1 (per global envs); 0 disables curriculum')
    parser.add_argument('--reward_switch_after_steps', type=int, default=0,
                        help='Global steps after which to switch reward to sparse (0 disables)')
    
    # Training
    parser.add_argument('--config', type=str, default='config/ogbench_config.yaml',
                        help='Path to RSL-RL config file')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--clip_actions', type=float, default=1.0,
                        help='Action clipping value')
    
    # Intervention / Teacher-Student
    parser.add_argument('--use_intervention', action='store_true', default=False,
                        help='Enable intervention wrapper (teacher-student)')
    parser.add_argument('--intervention_mode', type=str, default='agent', choices=['human', 'agent'],
                        help='Intervention mode: human teleop or agent teacher')
    parser.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs'],
                        help='Teacher type when mode=agent')
    parser.add_argument('--tolerance_type', type=str, default='angle', choices=['angle', 'l2'],
                        help='Intervention tolerance metric')
    parser.add_argument('--tolerance_value', type=float, default=30.0,
                        help='Tolerance threshold (deg for angle; abs for l2)')
    parser.add_argument('--hard_block_lethal', action='store_true', default=True,
                        help='Intervene if student would step into lethal cell')
    parser.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    parser.add_argument('--intervention_enable_after_steps', type=int, default=0,
                        help='Warmup steps per env before enabling interventions')
    
    # Logging
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='logs/rsl_rl',
                        help='Logging directory')
    
    # Rendering
    parser.add_argument('--render_during_training', action='store_true', default=False,
                        help='Show environment during training')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"🚀 RSL-RL Integrated Training on {args.env_name}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_short = args.env_name.replace('-v0', '').replace('maze', '')
        reward_suffix = f"{args.reward_type}"
        args.experiment_name = f"{env_short}_{reward_suffix}_{timestamp}"
    
    # Load RSL-RL configuration
    print(f"📄 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments  
    train_cfg = config  # Pass full config to OnPolicyRunner
    train_cfg['experiment_name'] = args.experiment_name
    
    # Calculate max iterations from total timesteps
    num_steps_per_env = train_cfg['num_steps_per_env']
    max_iterations = args.total_timesteps // (args.num_envs * num_steps_per_env)
    train_cfg['max_iterations'] = max_iterations
    
    print(f"📊 Training Configuration:")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Max iterations: {max_iterations:,}")
    print(f"   Steps per env: {num_steps_per_env}")
    print(f"   Number of envs: {args.num_envs}")
    print(f"   Reward type: {args.reward_type}")
    print(f"   Learning rate: {train_cfg['algorithm']['learning_rate']}")
    
    # Create environment
    print(f"🌍 Creating environment...")
    env = create_env(
        env_name=args.env_name,
        num_envs=args.num_envs,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        use_subgoal_shaping=args.use_subgoal_shaping,
        subgoal_shaping_coef=args.subgoal_shaping_coef,
        subgoal_shaping_gamma=args.subgoal_shaping_gamma,
        curriculum_stage1_steps=args.curriculum_stage1_steps,
        reward_switch_after_steps=args.reward_switch_after_steps,
        clip_actions=args.clip_actions,
        render_mode='human' if args.render_during_training else None,
        max_episode_steps=args.max_episode_steps,
        use_intervention=args.use_intervention,
        intervention_mode=args.intervention_mode,
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
    )
    
    print(f"✅ Environment created successfully")
    print(f"   Number of environments: {env.num_envs}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Max episode steps: {env.max_episode_length}")
    print(f"   Num obs: {env.num_obs}")
    print(f"   Num privileged obs: {env.num_privileged_obs}")
    
    # Create log directory
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Log directory: {log_dir}")
    
    # Create RSL-RL OnPolicyRunner
    print(f"🏃 Creating RSL-RL OnPolicyRunner...")
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=device)
    
    print(f"✅ OnPolicyRunner created successfully")
    print(f"   Algorithm: {train_cfg['algorithm'].get('class_name', 'PPO')}")
    print(f"   Policy: {train_cfg['policy'].get('class_name', 'ActorCritic')}")
    print(f"   Logger: {train_cfg.get('logger', 'wandb')}")
    
    # Start training
    print(f"🏋️  Starting RSL-RL training...")
    print(f"=" * 80)
    
    try:
        # Train using RSL-RL's runner
        runner.learn(num_learning_iterations=max_iterations)
        
        print(f"=" * 80)
        print(f"✅ Training completed successfully!")
        print(f"   Experiment: {args.experiment_name}")
        print(f"   Log directory: {log_dir}")
        print(f"   Total iterations: {max_iterations}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    finally:
        # Clean up
        env.close()
        print(f"🧹 Environment closed")


if __name__ == "__main__":
    main()
