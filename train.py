#!/usr/bin/env python3
"""
RSL-RL PPO Training Script for OGBench Point Locomotion
=====================================================

This script trains an agent using RSL-RL's PPO algorithm on OGBench point maze environments.
The environment wrapper is designed to be reusable with FastTD3 and other algorithms later.

Usage:
    python train.py --env_name pointmaze-medium-v0 --num_envs 1024 --total_timesteps 1000000
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import wandb
from pathlib import Path

# Add the fasttd3 directory to path for imports
sys.path.append('fasttd3/fast_sac')

# Import RSL-RL components
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

# Import ogbench to register environments
import ogbench

# Import our OGBench environment wrapper
from fasttd3.fast_sac.environments.ogbench_env import OGBenchEnv


class OGBenchRSLRLVecEnv(VecEnv):
    """Proper RSL-RL VecEnv wrapper for OGBench environments using TensorDict."""
    
    def __init__(self, env: OGBenchEnv, cfg: dict = None):
        self.env = env
        self.cfg = cfg or {}
        
        # RSL-RL VecEnv required attributes (exactly as per VecEnv abstract class)
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_steps
        self.device = env.sim_device
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state
        self._last_obs = None
        
        # Initialize observations
        self.reset()
        
    def get_observations(self) -> TensorDict:
        """Return current observations as TensorDict - required by RSL-RL."""
        if self._last_obs is None:
            self.reset()
        return self._last_obs
    
    def reset(self) -> TensorDict:
        """Reset environment and return observations as TensorDict."""
        # Get raw observations from OGBench environment
        raw_obs = self.env.reset()  # Shape: [num_envs, obs_dim]
        
        # Convert to TensorDict with proper structure for RSL-RL
        # RSL-RL expects observations grouped by purpose
        self._last_obs = TensorDict({
            "policy": raw_obs,  # Observations for policy network
            # Could add "critic" group if we had privileged observations
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step environment - RSL-RL signature with TensorDict observations."""
        # Step the underlying environment
        raw_obs, rewards, dones, infos = self.env.step(actions)
        
        # Update episode lengths
        self.episode_length_buf += 1
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": raw_obs,  # Policy observations
        }, batch_size=[self.num_envs], device=self.device)
        
        # Store for get_observations()
        self._last_obs = obs_tensordict
        
        # Ensure proper tensor shapes for RSL-RL
        if rewards.dim() == 1:
            rewards = rewards  # Keep as [num_envs] - RSL-RL expects this shape
        if dones.dim() == 1:
            dones = dones  # Keep as [num_envs] - RSL-RL expects this shape
            
        # Create extras dict with required RSL-RL fields
        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        extras = {
            "time_outs": time_outs,  # Required by RSL-RL for bootstrapping
            "log": {}  # Additional logging info
        }
        
        # Reset episode lengths for done/timeout environments
        reset_mask = dones.bool() | time_outs.bool()
        self.episode_length_buf[reset_mask] = 0
        
        return obs_tensordict, rewards, dones, extras
    
    def close(self):
        """Close the environment."""
        self.env.close()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RSL-RL PPO Training for OGBench')
    
    # Environment arguments
    parser.add_argument('--env_name', type=str, default='pointmaze-medium-v0',
                        help='OGBench environment name')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for policy and value networks')
    parser.add_argument('--num_learning_epochs', type=int, default=5,
                        help='Number of learning epochs per update')
    parser.add_argument('--num_mini_batches', type=int, default=4,
                        help='Number of mini-batches per update')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value_loss_coef', type=float, default=1.0,
                        help='Value loss coefficient')
    
    # Network arguments
    parser.add_argument('--hidden_dims', type=str, default='256,256',
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (in policy updates)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Model saving interval (in policy updates)')
    parser.add_argument('--eval_interval', type=int, default=250,
                        help='Evaluation interval (in policy updates)')
    parser.add_argument('--max_iterations', type=int, default=10000,
                        help='Maximum number of policy updates')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ogbench-rsl-rl',
                        help='Wandb project name')
    
    return parser.parse_args()


def create_rsl_rl_configs(args, env):
    """Create RSL-RL policy and algorithm configurations."""
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # Policy configuration for ActorCritic
    policy_cfg = {
        'init_noise_std': 1.0,
        'actor_hidden_dims': hidden_dims,
        'critic_hidden_dims': hidden_dims,
        'activation': args.activation,
    }
    
    # Algorithm configuration for PPO
    alg_cfg = {
        'value_loss_coef': args.value_loss_coef,
        'use_clipped_value_loss': True,
        'clip_param': args.clip_param,
        'entropy_coef': args.entropy_coef,
        'num_learning_epochs': args.num_learning_epochs,
        'num_mini_batches': args.num_mini_batches,
        'learning_rate': args.learning_rate,
        'schedule': 'adaptive',
        'gamma': args.gamma,
        'lam': args.lam,
        'desired_kl': 0.01,
        'max_grad_norm': 1.0,
    }
    
    # Runner configuration for OnPolicyRunner
    runner_cfg = {
        'num_steps_per_env': 1024,  # Number of steps per environment per rollout
        'max_iterations': args.max_iterations,
        'save_interval': args.save_interval,
        'log_interval': args.log_interval,
        'experiment_name': args.experiment_name,
        'logger': 'wandb' if args.use_wandb else 'tensorboard',
        'wandb_project': args.wandb_project if args.use_wandb else None,
    }
    
    return policy_cfg, alg_cfg, runner_cfg





def main():
    """Main training function using RSL-RL's proper pipeline."""
    args = get_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"ogbench_{args.env_name}_rslrl_ppo_{timestamp}"
    
    print(f"🚀 Starting RSL-RL PPO training: {args.experiment_name}")
    
    # Create environment
    print(f"Creating environment: {args.env_name} with {args.num_envs} parallel environments")
    try:
        base_env = OGBenchEnv(
            env_name=args.env_name,
            num_envs=args.num_envs,
            device=device
        )
        env = OGBenchRSLRLVecEnv(base_env, cfg={})
        print(f"✓ Environment created successfully")
        print(f"  - Envs: {env.num_envs}")
        print(f"  - Observations: {base_env.num_obs}")
        print(f"  - Actions: {env.num_actions}")
        print(f"  - Max episode length: {env.max_episode_length}")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        print("Try reducing --num_envs (e.g., --num_envs 1) or check your MuJoCo installation")
        return
    
    # Create RSL-RL configurations
    policy_cfg, alg_cfg, runner_cfg = create_rsl_rl_configs(args, env)
    
    # Create policy (ActorCritic)
    print("Creating ActorCritic policy...")
    try:
        # Get dummy observations for ActorCritic initialization
        dummy_obs = env.get_observations()
        
        print(f"🔍 Debugging RSL-RL observation requirements:")
        print(f"  - Original obs shape: {dummy_obs.shape}")
        print(f"  - Original obs: {dummy_obs}")
        
        # Let's try MULTIPLE approaches systematically until one works
        approaches = []
        
        # APPROACH 1: Column-wise feature selection (most likely correct)
        # obs_groups might index columns, not rows: obs[:, [0,1]] instead of obs[[0,1]]
        approaches.append({
            "name": "Column indexing",
            "obs": dummy_obs.cpu(),  # [1, 2] 
            "obs_groups": {"policy": [0, 1]},  # Select columns 0, 1
            "expected": "obs[:, [0,1]] -> [1, 2]"
        })
        
        # APPROACH 2: Dictionary structure 
        # Maybe RSL-RL expects obs as dict, not tensor
        approaches.append({
            "name": "Dictionary structure", 
            "obs": {"policy": dummy_obs.cpu()},  # Dict with policy key
            "obs_groups": {"policy": [0, 1]},
            "expected": "obs['policy'][:, [0,1]] -> [1, 2]"
        })
        
        # APPROACH 3: Expanded multi-env structure
        # Force multiple environments structure
        dummy_multi = dummy_obs.repeat(max(2, env.num_envs), 1)  # [2, 2] or [num_envs, 2]
        approaches.append({
            "name": "Multi-env structure",
            "obs": dummy_multi.cpu(),
            "obs_groups": {"policy": [0, 1]}, 
            "expected": f"obs[:, [0,1]] -> [{dummy_multi.shape[0]}, 2]"
        })
        
        # APPROACH 4: Flattened then grouped
        # Maybe obs should be [num_envs, all_features] then grouped
        approaches.append({
            "name": "Flattened structure",
            "obs": dummy_obs.cpu().view(1, -1),  # [1, 2] -> [1, 2]
            "obs_groups": {"policy": list(range(env.num_obs))},
            "expected": f"obs[:, 0:2] -> [1, {env.num_obs}]"
        })
        
        # Test each approach
        policy = None
        for i, approach in enumerate(approaches):
            print(f"\n📝 APPROACH {i+1}: {approach['name']}")
            print(f"  - Obs type: {type(approach['obs'])}")
            if isinstance(approach['obs'], torch.Tensor):
                print(f"  - Obs shape: {approach['obs'].shape}")
            print(f"  - Obs groups: {approach['obs_groups']}")
            print(f"  - Expected result: {approach['expected']}")
            
            # Test indexing behavior
            try:
                test_obs = approach['obs']
                test_groups = approach['obs_groups']['policy']
                
                if isinstance(test_obs, dict):
                    test_result = test_obs['policy'][:, test_groups] if len(test_groups) > 1 else test_obs['policy']
                else:
                    # Try column indexing first
                    test_result = test_obs[:, test_groups]
                    
                print(f"  - Test result shape: {test_result.shape}")
                print(f"  - Test result dims: {len(test_result.shape)}")
                
                if len(test_result.shape) == 2:
                    print(f"  ✓ Correct 2D shape - trying ActorCritic...")
                    
                    policy = ActorCritic(
                        obs=approach['obs'],
                        obs_groups=approach['obs_groups'],
                        num_actions=env.num_actions,
                        **policy_cfg
                    ).to(device)
                    
                    print(f"  🎉 SUCCESS with approach: {approach['name']}")
                    break
                else:
                    print(f"  ❌ Wrong dimensions: {len(test_result.shape)}")
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        if policy is None:
            print(f"\n💡 All approaches failed. Let me try one more systematic approach...")
            print(f"Let me examine what RSL-RL actually expects by looking at the source...")
            
            # Last resort: try to understand the exact structure RSL-RL wants
            print("📚 Based on assertion 'len(obs[obs_group].shape) == 2':")
            print("   This means obs[obs_group] must return a 2D tensor")
            print("   Let me try obs_groups as feature indices for column selection:")
            
            final_obs = dummy_obs.cpu()  # [1, 2]
            final_groups = {"policy": slice(0, env.num_obs)}  # Use slice instead of list
            
            print(f"  - Final attempt with slice: obs[:, {final_groups['policy']}]")
            test_slice = final_obs[:, final_groups['policy']]
            print(f"  - Slice result: {test_slice.shape}")
            
            policy = ActorCritic(
                obs=final_obs,
                obs_groups=final_groups,
                num_actions=env.num_actions,
                **policy_cfg
            ).to(device)
        
        print(f"✓ ActorCritic created successfully")
        print(f"  - Actor obs: {env.num_obs}")
        print(f"  - Critic obs: {env.num_obs}")
        print(f"  - Actions: {env.num_actions}")
        print(f"  - Policy device: {next(policy.parameters()).device}")
        
    except Exception as e:
        print(f"❌ ActorCritic creation failed: {e}")
        print("This suggests RSL-RL has specific requirements not met by our setup.")
        print("Debug info:")
        print(f"  - Target device: {device}")
        print(f"  - Obs shape: {dummy_obs.shape if 'dummy_obs' in locals() else 'N/A'}")
        print(f"  - Obs device: {dummy_obs.device if 'dummy_obs' in locals() else 'N/A'}")
        print(f"  - CPU obs shape: {dummy_obs_cpu.shape if 'dummy_obs_cpu' in locals() else 'N/A'}")
        print(f"  - Environment info: {env.num_envs} envs, {env.num_obs} obs, {env.num_actions} actions")
        import traceback
        traceback.print_exc()
        return
    
    # Create PPO algorithm
    print("Creating PPO algorithm...")
    try:
        ppo = PPO(policy, device=device, **alg_cfg)
        print("✓ PPO algorithm created successfully")
        
    except Exception as e:
        print(f"❌ PPO creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create OnPolicyRunner
    print("Creating OnPolicyRunner...")
    try:
        runner = OnPolicyRunner(env, ppo, **runner_cfg)
        print("✓ OnPolicyRunner created successfully")
        
    except Exception as e:
        print(f"❌ OnPolicyRunner creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize wandb logging through RSL-RL's system
    if args.use_wandb:
        print(f"🔗 Initializing wandb logging: {args.wandb_project}")
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config={
                    **vars(args),
                    'policy_cfg': policy_cfg,
                    'alg_cfg': alg_cfg,
                    'runner_cfg': runner_cfg,
                    'env_info': {
                        'num_envs': env.num_envs,
                        'num_obs': env.num_obs,
                        'num_actions': env.num_actions,
                        'max_episode_length': env.max_episode_length,
                    }
                }
            )
            # Set wandb logger in runner if available
            if hasattr(runner, 'logger') and hasattr(runner.logger, 'wandb'):
                runner.logger.wandb = wandb
            print("✓ Wandb logging initialized")
            
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
            print("Continuing without wandb logging...")
    
    # Training loop using RSL-RL's OnPolicyRunner
    print("\n🎯 Starting training with RSL-RL OnPolicyRunner...")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Steps per rollout: {runner_cfg['num_steps_per_env']}")
    print(f"Total timesteps target: {args.total_timesteps}")
    print("-" * 80)
    
    start_time = datetime.now()
    
    try:
        for iteration in range(args.max_iterations):
            # Run one iteration of PPO training
            runner.run()
            
            # Calculate metrics
            elapsed_time = (datetime.now() - start_time).total_seconds()
            total_timesteps = iteration * env.num_envs * runner_cfg['num_steps_per_env']
            fps = total_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            # Logging
            if iteration % args.log_interval == 0:
                print(f"Iteration {iteration:6d} | "
                      f"Timesteps: {total_timesteps:8d} | "
                      f"FPS: {fps:6.0f} | "
                      f"Time: {elapsed_time:.1f}s")
                
                if args.use_wandb and 'wandb' in locals():
                    wandb.log({
                        'iteration': iteration,
                        'total_timesteps': total_timesteps,
                        'fps': fps,
                        'elapsed_time': elapsed_time,
                    })
            
            # Check if we've reached target timesteps
            if total_timesteps >= args.total_timesteps:
                print(f"🎯 Reached target timesteps: {args.total_timesteps}")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    print("\n💾 Saving final model...")
    save_path = Path("models") / args.experiment_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_path = save_path / "final_model.pt"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'iteration': iteration if 'iteration' in locals() else 0,
        'args': vars(args),
        'policy_cfg': policy_cfg,
        'alg_cfg': alg_cfg,
    }, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    env.close()
    
    if args.use_wandb and 'wandb' in locals():
        wandb.finish()
        print("✓ Wandb session finished")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n🎉 Training completed in {elapsed:.1f}s!")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()


