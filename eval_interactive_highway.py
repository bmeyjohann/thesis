#!/usr/bin/env python3
"""
Interactive evaluation for HighwayEnv with optional human interventions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

FAST_SAC_PATH = Path(__file__).resolve().parent / "fasttd3" / "fast_sac"
HIGHWAY_PATH = Path(__file__).resolve().parent / "HighwayEnv"
if FAST_SAC_PATH.exists():
    sys.path.append(str(FAST_SAC_PATH))
if HIGHWAY_PATH.exists():
    sys.path.append(str(HIGHWAY_PATH))

from fast_sac import Actor  # noqa: E402
from fast_sac_utils import EmpiricalNormalization  # noqa: E402

import highway_env  # noqa: F401  # Registers gymnasium environments.  # noqa: E402
from highway_env import utils as hw_utils  # noqa: E402

try:  # noqa: E402
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for HighwayEnv evaluation") from exc

try:  # noqa: E402
    import pygame
except ImportError as exc:  # pragma: no cover
    raise ImportError("pygame is required for interactive evaluation") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive evaluation for HighwayEnv policies.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--expert_model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to a training args.json to mirror config.")
    parser.add_argument("--env_name", type=str, default="highway-v0")
    parser.add_argument("--controller", type=str, default="policy",
                        choices=("policy", "random", "human", "heuristic", "expert_model"))
    parser.add_argument("--intervention_mode", type=str, default="none",
                        choices=("none", "human"),
                        help="Enable human intervention overlay on top of policy controller.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render_mode", type=str, default="human", choices=("human", "rgb_array"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--device_rank", type=int, default=0)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--init_scale", type=float, default=0.01)

    parser.add_argument("--lanes_count", type=int, default=None)
    parser.add_argument("--traffic_vehicles", type=int, default=None)
    parser.add_argument("--obs_vehicles", type=int, default=5)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--collision_reward", type=float, default=None)
    parser.add_argument("--right_lane_reward", type=float, default=None)
    parser.add_argument("--high_speed_reward", type=float, default=None)
    parser.add_argument("--lane_change_reward", type=float, default=None)
    parser.add_argument("--reward_speed_min", type=float, default=None)
    parser.add_argument("--reward_speed_max", type=float, default=None)
    parser.add_argument("--normalize_reward", action="store_true", default=None)
    parser.add_argument("--no_normalize_reward", dest="normalize_reward", action="store_false")
    parser.add_argument("--offroad_terminal", action="store_true", default=None)
    parser.add_argument("--no_offroad_terminal", dest="offroad_terminal", action="store_false")

    parser.add_argument("--heuristic_target_speed", type=float, default=25.0)
    parser.add_argument("--heuristic_safe_distance", type=float, default=20.0)
    return parser.parse_args()


def _maybe_override(args: argparse.Namespace, key: str, value: Any) -> None:
    if value is None:
        return
    if getattr(args, key, None) is None:
        setattr(args, key, value)


def _load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _guess_args_path(model_path: str) -> Optional[str]:
    model = Path(model_path)
    run_name = model.parent.name
    args_path = Path("logs") / "fast_sac_highway" / run_name / "args.json"
    return str(args_path) if args_path.exists() else None


def _coerce_args_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return {k: v for k, v in vars(raw).items() if not k.startswith("_")}
    except TypeError:
        return {}


def apply_config_defaults(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    for key in (
        "env_name",
        "lanes_count",
        "obs_vehicles",
        "duration",
        "collision_reward",
        "right_lane_reward",
        "high_speed_reward",
        "lane_change_reward",
        "reward_speed_min",
        "reward_speed_max",
        "normalize_reward",
        "offroad_terminal",
        "actor_hidden_dim",
        "init_scale",
    ):
        if key in cfg:
            _maybe_override(args, key, cfg[key])
    if "traffic_vehicles" in cfg:
        _maybe_override(args, "traffic_vehicles", cfg["traffic_vehicles"])
    if "vehicles_count" in cfg:
        _maybe_override(args, "traffic_vehicles", cfg["vehicles_count"])
    if "observation" in cfg and isinstance(cfg["observation"], dict):
        obs_cfg = cfg["observation"]
        if "vehicles_count" in obs_cfg:
            _maybe_override(args, "obs_vehicles", obs_cfg["vehicles_count"])
    if "reward_speed_range" in cfg and isinstance(cfg["reward_speed_range"], (list, tuple)):
        speed_range = cfg["reward_speed_range"]
        if len(speed_range) >= 2:
            _maybe_override(args, "reward_speed_min", speed_range[0])
            _maybe_override(args, "reward_speed_max", speed_range[1])


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args.config_path:
        cfg = _load_json_config(args.config_path)
    elif args.model_path:
        guess = _guess_args_path(args.model_path)
        if guess is not None:
            cfg = _load_json_config(guess)

    if args.model_path and os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        cfg = {**_coerce_args_dict(ckpt.get("args")), **cfg}

    return cfg


def build_env_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": int(args.obs_vehicles),
        },
        "action": {"type": "ContinuousAction"},
        "lanes_count": int(args.lanes_count),
        "vehicles_count": int(args.traffic_vehicles),
        "duration": int(args.duration),
        "collision_reward": float(args.collision_reward),
        "right_lane_reward": float(args.right_lane_reward),
        "high_speed_reward": float(args.high_speed_reward),
        "lane_change_reward": float(args.lane_change_reward),
        "reward_speed_range": [float(args.reward_speed_min), float(args.reward_speed_max)],
        "normalize_reward": bool(args.normalize_reward),
        "offroad_terminal": bool(args.offroad_terminal),
    }


def make_env(env_name: str, config: Dict[str, Any], render_mode: str | None, max_steps: int) -> gym.Env:
    env_kwargs: Dict[str, Any] = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    if max_steps > 0:
        env_kwargs["max_episode_steps"] = max_steps
    env = gym.make(env_name, **env_kwargs)
    env.unwrapped.configure(config)
    env.reset()
    env = gym.wrappers.FlattenObservation(env)
    env.reset()
    return env


@dataclass
class LoadedPolicy:
    actor: Actor
    obs_normalizer: EmpiricalNormalization


def load_fastsac_policy(
    model_path: str,
    device: torch.device,
    args: argparse.Namespace,
    *,
    obs_dim: int,
    act_dim: int,
) -> LoadedPolicy:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    actor_hidden = int(train_args.get("actor_hidden_dim", args.actor_hidden_dim))
    init_scale = float(train_args.get("init_scale", args.init_scale))

    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        num_envs=1,
        init_scale=init_scale,
        hidden_dim=actor_hidden,
        device=device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    if checkpoint.get("obs_normalizer_state"):
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    obs_normalizer.eval()
    return LoadedPolicy(actor=actor, obs_normalizer=obs_normalizer)


def read_human_action(env: gym.Env, act_dim: int) -> np.ndarray:
    action_type = getattr(env.unwrapped, "action_type", None)
    if action_type is not None and hasattr(action_type, "last_action"):
        action = np.asarray(action_type.last_action, dtype=np.float32).reshape(-1)
    else:
        action = np.zeros(act_dim, dtype=np.float32)
    if action.size < act_dim:
        action = np.pad(action, (0, act_dim - action.size), mode="constant")
    elif action.size > act_dim:
        action = action[:act_dim]
    return action.astype(np.float32, copy=False)


def heuristic_action(env: gym.Env, target_speed: float, safe_distance: float, act_dim: int) -> np.ndarray:
    vehicle = env.unwrapped.vehicle
    road = env.unwrapped.road
    accel = np.clip((target_speed - vehicle.speed) * 0.1, -1.0, 1.0)
    try:
        front, _ = road.neighbour_vehicles(vehicle)
    except Exception:
        front = None
    if front is not None:
        distance = float(vehicle.lane_distance_to(front))
        if distance < safe_distance:
            accel = -1.0
        elif distance < safe_distance * 2.0:
            accel = min(accel, 0.0)

    steering = 0.0
    try:
        lane = road.network.get_lane(vehicle.lane_index)
        s, _ = lane.local_coordinates(vehicle.position)
        lane_heading = lane.heading_at(s)
        heading_error = hw_utils.wrap_to_pi(lane_heading - vehicle.heading)
        steering = float(np.clip(heading_error / (np.pi / 4), -1.0, 1.0))
    except Exception:
        steering = 0.0

    if act_dim == 1:
        return np.array([accel], dtype=np.float32)
    return np.array([accel, steering], dtype=np.float32)


def select_device(args: argparse.Namespace) -> torch.device:
    if not args.cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.device_rank}")
    if torch.backends.mps.is_available():
        return torch.device(f"mps:{args.device_rank}")
    raise RuntimeError("No GPU available")


def main() -> int:
    args = parse_args()
    device = select_device(args)

    cfg = resolve_config(args)

    base_env = gym.make(args.env_name)
    try:
        base_config = base_env.unwrapped.default_config()
    finally:
        try:
            base_env.close()
        except Exception:
            pass

    cfg = {**base_config, **cfg}
    apply_config_defaults(args, cfg)

    def _fill_from_base() -> None:
        apply_config_defaults(args, base_config)

    missing = [
        k
        for k in (
            "lanes_count",
            "traffic_vehicles",
            "obs_vehicles",
            "duration",
            "collision_reward",
            "right_lane_reward",
            "high_speed_reward",
            "lane_change_reward",
            "reward_speed_min",
            "reward_speed_max",
            "normalize_reward",
            "offroad_terminal",
        )
        if getattr(args, k) is None
    ]
    if missing:
        _fill_from_base()
        missing = [k for k in missing if getattr(args, k) is None]
    if missing:
        raise ValueError(f"Missing config values: {missing}. Provide --config_path or --model_path.")

    env_config = build_env_config(args)
    render_mode = None if args.headless else args.render_mode
    env = make_env(args.env_name, env_config, render_mode, args.max_episode_steps)

    policy = None
    expert_policy = None
    prefetched_obs, prefetched_info = env.reset(seed=args.seed)
    act_dim = int(np.prod(env.action_space.shape))
    obs_dim = int(np.prod(env.observation_space.shape))

    if args.model_path and args.controller == "policy":
        policy = load_fastsac_policy(args.model_path, device, args, obs_dim=obs_dim, act_dim=act_dim)

    if args.controller == "expert_model":
        if not args.expert_model_path:
            raise ValueError("controller='expert_model' requires --expert_model_path.")
        expert_policy = load_fastsac_policy(
            args.expert_model_path,
            device,
            args,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )

    teleop = None
    if args.controller == "human" or args.intervention_mode == "human":
        print("🎮 Human control active – focus the HighwayEnv window and use arrow keys.")

    pygame.init()
    clock = pygame.time.Clock()
    current_fps = float(max(1, args.fps))
    if render_mode == "human" and not args.headless:
        try:
            env.render()
        except Exception:
            pass

    print("🚀 HighwayEnv Evaluation")
    print(f"   env={args.env_name} controller={args.controller} intervention={args.intervention_mode}")
    print(f"   render={render_mode} episodes={args.num_episodes} device={device}")
    print(f"   action_space={env.action_space}")

    episode = 0
    episode_rewards = []
    episode_lengths = []
    running = True
    step_idx = 0

    while running and (args.num_episodes == 0 or episode < args.num_episodes):
        if episode == 0:
            obs, info = prefetched_obs, prefetched_info
        else:
            obs, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            human_action = None
            if (args.controller == "human" or args.intervention_mode == "human") and render_mode == "human":
                try:
                    env.render()
                except Exception:
                    pass
                human_action = read_human_action(env, act_dim)

            action = None
            if args.controller == "random":
                action = env.action_space.sample()
            elif args.controller == "human":
                action = human_action if human_action is not None else np.zeros(act_dim, dtype=np.float32)
            elif args.controller == "heuristic":
                action = heuristic_action(env, args.heuristic_target_speed, args.heuristic_safe_distance, act_dim)
            elif args.controller == "expert_model":
                obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).view(1, -1)
                with torch.no_grad():
                    norm_obs = expert_policy.obs_normalizer(obs_tensor)
                    _, _, mean = expert_policy.actor(norm_obs)
                action = mean.squeeze(0).cpu().numpy()
            elif args.controller == "policy":
                if policy is None:
                    action = np.zeros(act_dim, dtype=np.float32)
                else:
                    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).view(1, -1)
                    with torch.no_grad():
                        norm_obs = policy.obs_normalizer(obs_tensor)
                    _, _, mean = policy.actor(norm_obs)
                action = mean.squeeze(0).cpu().numpy()
            else:
                action = env.action_space.sample()

            if args.intervention_mode == "human" and human_action is not None:
                if float(np.linalg.norm(human_action)) > 1e-3:
                    action = human_action

            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_len += 1
            step_idx += 1
            if render_mode == "human":
                env.render()
            if step_idx % 10 == 0:
                try:
                    vehicle = env.unwrapped.vehicle
                    on_road = bool(vehicle.on_road)
                    lane_index = vehicle.lane_index
                    lane_id = int(lane_index[2]) if lane_index is not None else -1
                    speed = float(getattr(vehicle, "speed", 0.0))
                except Exception:
                    on_road = False
                    lane_id = -1
                    speed = 0.0
                crashed = bool(info.get("crashed", False)) if isinstance(info, dict) else False
                print(
                    f"step={step_idx} reward={reward:.3f} ep_reward={ep_reward:.3f} "
                    f"speed={speed:.2f} lane={lane_id} on_road={int(on_road)} crashed={int(crashed)} "
                    f"action={np.round(action, 3)}"
                )
            clock.tick(current_fps)

        if not running:
            break
        episode += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        print(f"Episode {episode}: return={ep_reward:.2f} len={ep_len}")

    if episode_rewards:
        print(
            f"Summary: return_mean={np.mean(episode_rewards):.2f} "
            f"return_std={np.std(episode_rewards):.2f} "
            f"len_mean={np.mean(episode_lengths):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
