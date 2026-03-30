#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from safetygym_utils.controllers import build_human_controller
from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
)
from safetygym_utils.gamepad_web import GamepadWebServer
from safetygym_utils.env import clip_action_to_space, extract_goal_distance, make_safety_env
from safetygym_utils.rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from safetygym_utils.wrappers import RewardModeWrapper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live SafetyGym gamepad configuration tool")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--render_mode", type=str, default="pygame", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--viewer_fps", type=float, default=20.0)
    p.add_argument("--viewer_scale", type=float, default=1.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="none", choices=["sparse", "dense", "dense_plus_sparse", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--human_action_scale", type=float, default=1.0)
    p.add_argument("--controller_fps_limit", type=int, default=0)
    p.add_argument("--controller_overlay_hz", type=float, default=20.0)
    p.add_argument("--gamepad_mode", type=str, default="local", choices=["local", "connect"])
    p.add_argument("--gamepad_host", type=str, default="")
    p.add_argument("--gamepad_port", type=int, default=0)
    p.add_argument("--gamepad_cache_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CACHE_PATH))
    p.add_argument("--gamepad_reconnect_seconds", type=float, default=2.0)
    p.add_argument("--gamepad_config_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH))
    p.add_argument("--gamepad_use_saved_config", action="store_true", default=True)
    p.add_argument("--no_gamepad_use_saved_config", dest="gamepad_use_saved_config", action="store_false")
    p.add_argument("--gamepad_device_index", type=int, default=0)
    p.add_argument("--web_port", type=int, default=8992)
    p.add_argument("--fps", type=int, default=30)
    return p


def _runtime_provider(runtime: dict[str, Any]):
    def _inner() -> dict[str, Any]:
        return {
            "summary": runtime.get("summary", "waiting"),
            "values": {
                "episode": runtime.get("episode", 0),
                "episode_length": runtime.get("episode_length", 0),
                "episode_return": f"{float(runtime.get('episode_return', 0.0)):.3f}",
                "episode_cost": f"{float(runtime.get('episode_cost', 0.0)):.3f}",
                "last_action": np.array2string(np.asarray(runtime.get("last_action", []), dtype=np.float32), precision=3),
                "final_distance": f"{float(runtime.get('final_distance', 0.0)):.3f}",
            },
        }
    return _inner


def main() -> int:
    args = build_parser().parse_args()
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        max_episode_steps=args.max_episode_steps,
        surface_mode=args.surface_mode,
        car_wheel_command_limit=args.car_wheel_command_limit,
        car_force_scale=args.car_force_scale,
        seed=args.seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
    )
    act_dim = int(np.prod(env.action_space.shape))
    controller = build_human_controller(
        input_device="gamepad",
        action_dim=act_dim,
        env_name=args.env_name,
        action_scale=float(args.human_action_scale),
        wheel_command_limit=float(args.car_wheel_command_limit),
        overlay_fps_limit=int(args.controller_fps_limit),
        overlay_draw_hz=float(args.controller_overlay_hz),
        gamepad_mode=str(args.gamepad_mode),
        gamepad_host=str(args.gamepad_host),
        gamepad_port=int(args.gamepad_port or DEFAULT_SAFETY_GAMEPAD_PORT),
        gamepad_cache_path=args.gamepad_cache_path,
        gamepad_reconnect_seconds=float(args.gamepad_reconnect_seconds),
        gamepad_config_path=args.gamepad_config_path,
        gamepad_use_saved_config=bool(args.gamepad_use_saved_config),
        gamepad_device_index=int(args.gamepad_device_index),
        prefer_separate_keyboard_window=wants_external_viewer(args.render_mode),
    )
    runtime = {
        "summary": "running",
        "episode": 1,
        "episode_length": 0,
        "episode_return": 0.0,
        "episode_cost": 0.0,
        "last_action": np.zeros((act_dim,), dtype=np.float32),
        "final_distance": 0.0,
    }
    web = GamepadWebServer(
        controller=controller,
        config_path=args.gamepad_config_path,
        port=int(args.web_port),
        runtime_provider=_runtime_provider(runtime),
    )
    web.start()
    print(f"gamepad web ui listening at {web.url()}", flush=True)
    print(f"gamepad profile path: {args.gamepad_config_path}", flush=True)
    viewer = build_external_viewer(
        render_mode=args.render_mode,
        title=f"SafetyGym Config {args.env_name}",
        draw_hz=float(args.viewer_fps),
        scale=float(args.viewer_scale),
    ) if wants_external_viewer(args.render_mode) else None

    obs, _ = env.reset(seed=args.seed)
    if viewer is not None:
        viewer.draw_env(env)
    episode = 1
    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0
    try:
        while True:
            action = controller.get_action().astype(np.float32)
            action = clip_action_to_space(action, env.action_space)
            runtime["last_action"] = action.copy()
            next_obs, reward, cost, terminated, truncated, _info = env.step(action)
            if viewer is not None:
                viewer.draw_env(env)
            _ = next_obs, obs
            ep_ret += float(reward)
            ep_cost += float(cost)
            ep_len += 1
            runtime["episode"] = episode
            runtime["episode_length"] = ep_len
            runtime["episode_return"] = ep_ret
            runtime["episode_cost"] = ep_cost
            runtime["final_distance"] = float(extract_goal_distance(env))
            if terminated or truncated:
                print(
                    f"episode={episode} return={ep_ret:.3f} cost={ep_cost:.3f} len={ep_len} final_dist={runtime['final_distance']:.3f}",
                    flush=True,
                )
                episode += 1
                obs, _ = env.reset(seed=args.seed + episode)
                if viewer is not None:
                    viewer.draw_env(env)
                ep_ret = 0.0
                ep_cost = 0.0
                ep_len = 0
            else:
                obs = next_obs
            if int(args.fps) > 0:
                time.sleep(1.0 / float(args.fps))
    except KeyboardInterrupt:
        pass
    finally:
        web.close()
        controller.close()
        if viewer is not None:
            viewer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
