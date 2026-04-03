#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from safetygym_utils.controllers import build_human_controller
from safetygym_utils.dataset_io import (
    DEFAULT_SAFETYGYM_DATASET_DIR,
    build_dataset_path,
    save_transition_dataset,
)
from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
)
from safetygym_utils.env import clip_action_to_space, make_safety_env
from safetygym_utils.rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from safetygym_utils.wrappers import RewardModeWrapper


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect reusable SafetyGym human demo datasets")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--render_mode", type=str, default="pygame", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--viewer_fps", type=float, default=20.0)
    p.add_argument("--viewer_scale", type=float, default=1.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="sparse", choices=["sparse", "dense", "dense_plus_sparse", "native", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--human_input_device", type=str, default="gamepad", choices=["keyboard", "gamepad"])
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
    p.add_argument("--dataset_dir", type=str, default=str(DEFAULT_SAFETYGYM_DATASET_DIR))
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--dataset_label", type=str, default="human_demo")
    return p


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
        input_device=str(args.human_input_device),
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

    obs_rows = []
    action_rows = []
    next_obs_rows = []
    reward_rows = []
    done_rows = []
    trunc_rows = []
    cost_rows = []
    student_rows = []
    intervened_rows = []
    viewer = build_external_viewer(
        render_mode=args.render_mode,
        title=f"SafetyGym Demo Collect {args.env_name}",
        draw_hz=float(args.viewer_fps),
        scale=float(args.viewer_scale),
    ) if wants_external_viewer(args.render_mode) else None

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if viewer is not None:
        viewer.draw_env(env)
    episodes = 0
    steps = 0
    try:
        while episodes < int(args.num_episodes):
            action = controller.get_action().astype(np.float32)
            action = clip_action_to_space(action, env.action_space)
            next_obs, reward, cost, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            if viewer is not None:
                viewer.draw_env(env)
            obs_rows.append(obs.copy())
            action_rows.append(action.copy())
            next_obs_rows.append(next_obs.copy())
            reward_rows.append(float(reward))
            done_rows.append(bool(terminated or truncated))
            trunc_rows.append(bool(truncated))
            cost_rows.append(float(cost))
            student_rows.append(np.zeros_like(action, dtype=np.float32))
            intervened_rows.append(True)
            steps += 1
            if terminated or truncated:
                episodes += 1
                print(f"episode={episodes} collected_steps={steps}", flush=True)
                obs, _ = env.reset(seed=args.seed + episodes)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                if viewer is not None:
                    viewer.draw_env(env)
            else:
                obs = next_obs
            if int(args.fps) > 0:
                time.sleep(1.0 / float(args.fps))
    except KeyboardInterrupt:
        print("Interrupted; saving collected data so far.", flush=True)
    finally:
        controller.close()
        if viewer is not None:
            viewer.close()
        env.close()

    if not obs_rows:
        print("No transitions collected; nothing saved.", flush=True)
        return 1

    dataset_path = (
        build_dataset_path(env_name=args.env_name, dataset_dir=args.dataset_dir, label=args.dataset_label)
        if not str(args.dataset_path).strip()
        else args.dataset_path
    )
    metadata = {
        "env_name": args.env_name,
        "input_device": args.human_input_device,
        "surface_mode": args.surface_mode,
        "reward_mode": args.reward_mode,
        "num_episodes_requested": int(args.num_episodes),
        "num_episodes_collected": int(episodes),
        "num_steps_collected": int(len(obs_rows)),
    }
    path = save_transition_dataset(
        path=dataset_path,
        metadata=metadata,
        observations=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        next_observations=np.asarray(next_obs_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        truncations=np.asarray(trunc_rows, dtype=np.bool_),
        costs=np.asarray(cost_rows, dtype=np.float32),
        student_actions=np.asarray(student_rows, dtype=np.float32),
        teacher_intervened=np.asarray(intervened_rows, dtype=np.bool_),
    )
    print(json.dumps({"dataset_path": str(path), "rows": len(obs_rows), "episodes": episodes}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
