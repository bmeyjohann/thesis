#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from safetygym_utils.io import save_args_json
from safetygym_utils.minimal_train import _make_env_with_wrappers


class SafetyGymnasiumToGymnasium(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("cost", float(cost))
        return obs, float(reward), bool(terminated), bool(truncated), info


class WandbMetricCallback(BaseCallback):
    def __init__(self, *, wandb_run, log_freq: int = 2048):
        super().__init__()
        self.wandb_run = wandb_run
        self.log_freq = int(max(1, log_freq))

    def _on_step(self) -> bool:
        if self.wandb_run is None or (self.num_timesteps % self.log_freq) != 0:
            return True
        metrics: dict[str, float] = {"train/step": float(self.num_timesteps)}
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                metrics[f"sb3/{key}"] = float(value)
        self.wandb_run.log(metrics, step=int(self.num_timesteps))
        return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal PPO training for Safety-Gymnasium with thesis reward wrappers.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--vec_env", type=str, default="subproc", choices=["dummy", "subproc"])
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--net_arch", type=str, default="256,256")
    p.add_argument("--activation_fn", type=str, default="tanh", choices=["tanh", "relu"])
    p.add_argument("--normalize_obs", action="store_true", default=True)
    p.add_argument("--no_normalize_obs", dest="normalize_obs", action="store_false")
    p.add_argument("--normalize_reward", action="store_true", default=False)
    p.add_argument("--render_mode", type=str, default="none", choices=["human", "none"])
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument("--point_turn_gain", type=float, default=2.5)
    p.add_argument("--point_alignment_power", type=float, default=1.0)
    p.add_argument("--point_allow_backward", action="store_true", default=False)
    p.add_argument("--obs_mask_mode", type=str, default="none", choices=["none", "goal_only_lidar"])
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--terminate_on_goal", action="store_true", default=False)
    p.add_argument("--reseed_on_episode_reset", action="store_true", default=False)
    p.add_argument("--no_reseed_on_episode_reset", dest="reseed_on_episode_reset", action="store_false")
    p.add_argument("--reward_mode", type=str, default="dense_plus_sparse", choices=["sparse", "dense", "dense_plus_sparse", "potential_diff", "native", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=-0.001)
    p.add_argument("--cost_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--cost_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_scale", type=float, default=1.1)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_power", type=float, default=1.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="softplus", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.001)
    p.add_argument("--clearance_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--forward_reward_scale", type=float, default=0.0)
    p.add_argument("--backward_penalty_scale", type=float, default=0.0)
    p.add_argument("--heading_reward_scale", type=float, default=0.0)
    p.add_argument("--heading_positive_only", action="store_true", default=True)
    p.add_argument("--no_heading_positive_only", dest="heading_positive_only", action="store_false")
    p.add_argument("--adaptive_safety_curriculum", action="store_true", default=False)
    p.add_argument("--adaptive_safety_goal_target", type=float, default=1.0)
    p.add_argument("--adaptive_safety_window_episodes", type=int, default=10)
    p.add_argument("--adaptive_safety_step", type=float, default=0.05)
    p.add_argument("--adaptive_safety_init", type=float, default=0.0)
    p.add_argument("--adaptive_safety_min", type=float, default=0.0)
    p.add_argument("--adaptive_safety_max", type=float, default=1.0)
    p.add_argument("--save_interval", type=int, default=20_000)
    p.add_argument("--log_interval", type=int, default=2048)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="thesis-safetygym")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    return p


def _prepare_run_dirs(args) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.exp_name:
        args.exp_name = f"ppo_{args.env_name.replace('-', '_')}_{stamp}"
    log_dir = Path("logs") / "safetygym_ppo" / args.exp_name
    model_dir = Path("models") / "safetygym_ppo" / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(log_dir / "args.json", vars(args))
    save_args_json(model_dir / "args.json", vars(args))
    return log_dir, model_dir


def _make_single_env(args, seed: int):
    env = _make_env_with_wrappers(args=args, seed=seed, with_intervention=False, controller=None)
    env = SafetyGymnasiumToGymnasium(env)
    return Monitor(env)


def _activation_fn(name: str):
    import torch.nn as nn

    return {"tanh": nn.Tanh, "relu": nn.ReLU}[str(name).lower()]


def _maybe_init_wandb(args, log_dir: Path):
    if not bool(args.use_wandb):
        return None
    import wandb

    kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "mode": args.wandb_mode,
        "config": vars(args),
        "dir": str(log_dir),
        "name": args.wandb_run_name or args.exp_name,
        "group": args.wandb_group or None,
    }
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    return wandb.init(**kwargs)


def main() -> None:
    args = build_parser().parse_args()
    log_dir, model_dir = _prepare_run_dirs(args)
    wandb_run = _maybe_init_wandb(args, log_dir)

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and args.num_envs > 1 else DummyVecEnv
    env_fns = [
        (lambda rank=rank: _make_single_env(args, seed=int(args.seed) + rank))
        for rank in range(int(args.num_envs))
    ]
    env = vec_cls(env_fns)
    if args.normalize_obs or args.normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=bool(args.normalize_obs),
            norm_reward=bool(args.normalize_reward),
            gamma=float(args.gamma),
        )

    net_arch = [int(x.strip()) for x in str(args.net_arch).split(",") if x.strip()]
    policy_kwargs = {
        "net_arch": {"pi": net_arch, "vf": net_arch},
        "activation_fn": _activation_fn(args.activation_fn),
    }
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=int(args.seed),
        device=args.device,
        tensorboard_log=str(log_dir / "tb"),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        learning_rate=float(args.learning_rate),
        clip_range=float(args.clip_range),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        policy_kwargs=policy_kwargs,
    )
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, int(args.save_interval) // max(1, int(args.num_envs))),
            save_path=str(model_dir),
            name_prefix="ppo_step",
            save_vecnormalize=bool(args.normalize_obs or args.normalize_reward),
        ),
        WandbMetricCallback(wandb_run=wandb_run, log_freq=int(args.log_interval)),
    ]
    model.learn(total_timesteps=int(args.total_timesteps), callback=callbacks, progress_bar=False)
    model.save(str(model_dir / "final.zip"))
    if isinstance(env, VecNormalize):
        env.save(str(model_dir / "vecnormalize.pkl"))
    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
