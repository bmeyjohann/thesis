#!/usr/bin/env python3
"""
Evaluate a FastSAC HighwayEnv policy checkpoint.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

FAST_SAC_PATH = Path(__file__).resolve().parent / "fasttd3" / "fast_sac"
HIGHWAY_PATH = Path(__file__).resolve().parent / "HighwayEnv"
if FAST_SAC_PATH.exists():
    os.sys.path.append(str(FAST_SAC_PATH))
if HIGHWAY_PATH.exists():
    os.sys.path.append(str(HIGHWAY_PATH))

from fast_sac import Actor  # noqa: E402
from fast_sac_utils import EmpiricalNormalization  # noqa: E402

import highway_env  # noqa: F401  # Registers gymnasium environments.  # noqa: E402

try:  # noqa: E402
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - gymnasium expected in runtime env
    raise ImportError("gymnasium is required for HighwayEnv evaluation") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastSAC checkpoint on HighwayEnv.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="highway-v0")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--render_fps", type=int, default=15)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--device_rank", type=int, default=0)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument("--use_checkpoint_arch", action="store_true", default=True)
    parser.add_argument("--lanes_count", type=int, default=4)
    parser.add_argument("--traffic_vehicles", type=int, default=50)
    parser.add_argument("--obs_vehicles", type=int, default=15)
    parser.add_argument("--duration", type=int, default=40)
    parser.add_argument("--collision_reward", type=float, default=-2.0)
    parser.add_argument("--right_lane_reward", type=float, default=0.0)
    parser.add_argument("--high_speed_reward", type=float, default=1.0)
    parser.add_argument("--lane_change_reward", type=float, default=0.0)
    parser.add_argument("--reward_speed_min", type=float, default=20.0)
    parser.add_argument("--reward_speed_max", type=float, default=30.0)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--offroad_terminal", action="store_true", default=False)
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> Dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": args.obs_vehicles,
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": args.lanes_count,
        "vehicles_count": args.traffic_vehicles,
        "duration": args.duration,
        "collision_reward": args.collision_reward,
        "right_lane_reward": args.right_lane_reward,
        "high_speed_reward": args.high_speed_reward,
        "lane_change_reward": args.lane_change_reward,
        "reward_speed_range": [args.reward_speed_min, args.reward_speed_max],
        "normalize_reward": args.normalize_reward,
        "offroad_terminal": args.offroad_terminal,
    }


def make_env(env_name: str, config: Dict, render: bool) -> gym.Env:
    env = gym.make(env_name, render_mode="human" if render else None)
    env.unwrapped.configure(config)
    env = gym.wrappers.FlattenObservation(env)
    return env


def main() -> None:
    args = parse_args()

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise RuntimeError("No GPU available")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {}) if args.use_checkpoint_arch else {}
    actor_hidden_dim = int(ckpt_args.get("actor_hidden_dim", args.actor_hidden_dim))
    init_scale = float(ckpt_args.get("init_scale", args.init_scale))

    env_config = build_env_config(args)
    env = make_env(args.env_name, env_config, render=args.render)
    obs, info = env.reset(seed=args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=1,
        init_scale=init_scale,
        hidden_dim=actor_hidden_dim,
        device=device,
    )
    obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    actor.eval()
    obs_normalizer.eval()

    returns = []
    lengths = []
    crashes = []
    speeds = []

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_speed = 0.0
        ep_speed_count = 0
        ep_crash = 0

        while not done:
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                norm_obs = obs_normalizer(obs_tensor)
                _, _, action_mean = actor(norm_obs)
            action = action_mean.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_len += 1
            ep_speed += float(info.get("speed", 0.0))
            ep_speed_count += 1
            if info.get("crashed", False):
                ep_crash = 1
            if args.render:
                env.render()
                time.sleep(1.0 / max(1, args.render_fps))

        returns.append(ep_return)
        lengths.append(ep_len)
        crashes.append(ep_crash)
        speeds.append(ep_speed / max(1, ep_speed_count))
        print(
            f"Episode {ep + 1}: return={ep_return:.2f} len={ep_len} "
            f"crash={ep_crash} speed={speeds[-1]:.2f}"
        )

    print(
        "Summary: "
        f"return_mean={np.mean(returns):.2f} return_std={np.std(returns):.2f} "
        f"len_mean={np.mean(lengths):.2f} crash_rate={np.mean(crashes):.2f} "
        f"speed_mean={np.mean(speeds):.2f}"
    )


if __name__ == "__main__":
    main()
