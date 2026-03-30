#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ogbench_utils.env_wrappers_common import FixedResetSeedWrapper, TeleopRobotStateSyncWrapper
from ogbench_utils.env_wrappers_manip import build_ogbench_manip_wrapper
from ogbench_utils.intervention_wrappers import DirectTeleopWrapper
from ogbench_utils.manip_topdown import extract_manip_topdown_state
from ogbench_utils.manip_dataset_io import DEFAULT_MANIP_DATASET_DIR, build_dataset_path, save_transition_dataset
from ogbench_utils.vr_mapping_web import create_vr_source
from ogbench_utils.vr_teleop import (
    DEFAULT_VR_CACHE_PATH,
    DEFAULT_VR_MAPPING_PATH,
    DEFAULT_VR_PORT,
    VRManipActionMapper,
    VRManipMappingConfig,
    VRStatusPanel,
    VRTeleopInterface,
    apply_vr_mapping_profile,
)


class NativeViewer:
    def __init__(self):
        self.enabled = False

    def maybe_launch(self, env: gym.Env) -> None:
        launch_fn = getattr(env.unwrapped, "launch_passive_viewer", None)
        if not callable(launch_fn):
            return
        try:
            launch_fn(show_left_ui=False, show_right_ui=False)
            self.enabled = True
        except Exception as exc:
            print(f"native viewer warning: launch failed: {exc}", flush=True)

    def sync(self, env: gym.Env) -> None:
        if not self.enabled:
            return
        sync_fn = getattr(env.unwrapped, "sync_passive_viewer", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                self.enabled = False

    def close(self, env: gym.Env) -> None:
        close_fn = getattr(env.unwrapped, "close_passive_viewer", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect reusable OGBench manipulation demo datasets with calibrated VR teleop.")
    p.add_argument("--env_name", type=str, default="cube-single-v0")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--mujoco_gl", type=str, default="auto")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--physics_timestep", type=float, default=0.002)
    p.add_argument("--control_timestep", type=float, default=0.02)
    p.add_argument("--hold_targets_on_zero_action", action="store_true", default=False)
    p.add_argument("--noop_action_threshold", type=float, default=1e-6)
    p.add_argument("--disable_rotation", action="store_true", default=False)
    p.add_argument("--include_goal", action="store_true", default=True)
    p.add_argument("--no_include_goal", dest="include_goal", action="store_false")
    p.add_argument("--include_relative_cube_features", action="store_true", default=False)
    p.add_argument("--relative_only_obs", action="store_true", default=False)
    p.add_argument("--reward_type", type=str, default="sparse", choices=["sparse", "dense", "combined", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--teacher_target_mode", type=str, default="sequential", choices=["fixed", "sequential"])
    p.add_argument("--cube_success_tolerance", type=float, default=0.04)
    p.add_argument("--static_reset_seed", type=int, default=-1)
    p.add_argument("--dataset_dir", type=str, default=str(DEFAULT_MANIP_DATASET_DIR))
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--dataset_label", type=str, default="human_vr_demo")
    p.add_argument("--vr_mode", type=str, default="connect", choices=["connect", "listen"])
    p.add_argument("--vr_host", type=str, default="")
    p.add_argument("--vr_port", type=int, default=0)
    p.add_argument("--vr_cache_path", type=str, default=str(DEFAULT_VR_CACHE_PATH))
    p.add_argument("--vr_mapping_path", type=str, default=str(DEFAULT_VR_MAPPING_PATH))
    p.add_argument("--vr_reconnect_seconds", type=float, default=2.0)
    p.add_argument("--vr_use_saved_mapping", action="store_true", default=True)
    p.add_argument("--no_vr_use_saved_mapping", dest="vr_use_saved_mapping", action="store_false")
    p.add_argument("--vr_hand", type=str, default="right", choices=["left", "right"])
    p.add_argument("--vr_gate_button", type=str, default="grip")
    p.add_argument("--vr_gripper_mirror_toggle_button", type=str, default="none")
    p.add_argument("--vr_require_gate", action="store_true", default=True)
    p.add_argument("--no_vr_require_gate", dest="vr_require_gate", action="store_false")
    p.add_argument("--show_status_panel", action="store_true", default=True)
    p.add_argument("--no_show_status_panel", dest="show_status_panel", action="store_false")
    p.add_argument("--show_topdown_view", action="store_true", default=True)
    p.add_argument("--no_show_topdown_view", dest="show_topdown_view", action="store_false")
    return p.parse_args()


def _create_env(args: argparse.Namespace, teleop: VRTeleopInterface) -> gym.Env:
    if args.render_mode == "human":
        os.environ["MUJOCO_GL"] = "glfw" if args.mujoco_gl == "auto" else args.mujoco_gl
    elif args.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    import ogbench  # noqa: F401

    make_kwargs: dict[str, Any] = {
        "render_mode": None if args.render_mode == "human" else args.render_mode,
        "physics_timestep": float(args.physics_timestep),
        "control_timestep": float(args.control_timestep),
    }
    if int(args.max_episode_steps) > 0:
        make_kwargs["max_episode_steps"] = int(args.max_episode_steps)
    make_kwargs["hold_targets_on_zero_action"] = bool(args.hold_targets_on_zero_action)
    make_kwargs["noop_action_threshold"] = float(args.noop_action_threshold)
    make_kwargs["disable_rotation"] = bool(args.disable_rotation)
    env = gym.make(args.env_name, **make_kwargs)
    if int(args.static_reset_seed) >= 0:
        env = FixedResetSeedWrapper(env, reset_seed=int(args.static_reset_seed))
    env = build_ogbench_manip_wrapper(
        env_name=args.env_name,
        obs_mode="state",
        include_goal=bool(args.include_goal),
        include_relative_cube_features=bool(args.include_relative_cube_features),
        relative_only_obs=bool(args.relative_only_obs),
        reward_type=str(args.reward_type),
        dense_reward_scale=float(args.dense_reward_scale),
        step_penalty=float(args.step_penalty),
        disable_rotation=bool(args.disable_rotation),
        intervention_mode="none",
        teacher_target_mode=str(args.teacher_target_mode),
        cube_success_tolerance=float(args.cube_success_tolerance),
    )(env)
    env = TeleopRobotStateSyncWrapper(env, teleop)
    env = DirectTeleopWrapper(env, teleop)
    return env


def main() -> int:
    args = _parse_args()
    viewer = NativeViewer()
    source = create_vr_source(
        vr_mode=args.vr_mode,
        vr_host=args.vr_host,
        vr_port=args.vr_port,
        cache_path=args.vr_cache_path,
        reconnect_seconds=args.vr_reconnect_seconds,
    )
    print(source.banner_text(), flush=True)

    mapping_path = Path(args.vr_mapping_path)
    mapping_config = VRManipMappingConfig(
        hand=args.vr_hand,
        require_gate=bool(args.vr_require_gate),
        gate_button=args.vr_gate_button,
        gripper_mirror_toggle_button=args.vr_gripper_mirror_toggle_button,
    )
    if bool(args.vr_use_saved_mapping):
        mapping_config, loaded = apply_vr_mapping_profile(mapping_config, mapping_path)
        if loaded:
            print(f"loaded vr mapping profile from {mapping_path}", flush=True)
    mapper_action_dim = 4 if bool(args.disable_rotation) else 5
    teleop = VRTeleopInterface(
        source,
        VRManipActionMapper(action_dim=mapper_action_dim, config=mapping_config),
        return_none_when_idle=False,
        mapping_path=mapping_path,
    )
    status_panel = None
    if args.render_mode == "human" and (bool(args.show_status_panel) or bool(args.show_topdown_view)):
        try:
            status_panel = VRStatusPanel(
                title="VR Demo Collector",
                action_dim=mapper_action_dim,
                mapping_config=teleop.mapper.config,
                mapping_path=mapping_path,
                show_topdown=bool(args.show_topdown_view),
            )
        except Exception as exc:
            print(f"status panel unavailable: {exc}", flush=True)
            status_panel = None
    env = _create_env(args, teleop)
    if args.render_mode == "human":
        viewer.maybe_launch(env)

    rows_obs = []
    rows_actions = []
    rows_next_obs = []
    rows_rewards = []
    rows_dones = []
    rows_trunc = []

    obs, info = env.reset(seed=int(args.seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    viewer.sync(env)
    if status_panel is not None:
        status_panel.set_topdown_state(extract_manip_topdown_state(env, info if isinstance(info, dict) else None))
    fps_dt = 1.0 / max(1.0, float(args.fps))

    try:
        stop_requested = False
        for ep_idx in range(int(args.num_episodes)):
            if stop_requested:
                break
            ep_steps = 0
            while True:
                t0 = time.perf_counter()
                if status_panel is not None:
                    status_panel.set_snapshot(source.snapshot())
                    status_panel.set_topdown_state(extract_manip_topdown_state(env, info if isinstance(info, dict) else None))
                    _prev, next_requested, quit_requested, advance_requested, _fps_delta = status_panel.poll()
                    status_panel.draw()
                    if quit_requested:
                        print("collector quit requested from status panel", flush=True)
                        stop_requested = True
                        break
                next_obs, reward, terminated, truncated, info = env.step(None)
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                action = np.asarray(
                    info.get("human_action", np.zeros(env.action_space.shape, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                rows_obs.append(obs.copy())
                rows_actions.append(action.copy())
                rows_next_obs.append(next_obs.copy())
                rows_rewards.append(float(reward))
                rows_dones.append(bool(terminated))
                rows_trunc.append(bool(truncated))
                obs = next_obs
                ep_steps += 1
                viewer.sync(env)
                if terminated or truncated:
                    print(f"episode={ep_idx + 1}/{int(args.num_episodes)} steps={ep_steps}", flush=True)
                    obs, info = env.reset(seed=int(args.seed) + ep_idx + 1)
                    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                    viewer.sync(env)
                    if status_panel is not None:
                        status_panel.set_topdown_state(extract_manip_topdown_state(env, info if isinstance(info, dict) else None))
                    break
                elapsed = time.perf_counter() - t0
                if elapsed < fps_dt:
                    time.sleep(fps_dt - elapsed)
    finally:
        if status_panel is not None:
            status_panel.close()
        viewer.close(env)
        env.close()
        source.close()

    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser()
    else:
        dataset_path = build_dataset_path(
            env_name=args.env_name,
            dataset_dir=args.dataset_dir,
            label=args.dataset_label,
        )
    metadata = {
        "env_name": str(args.env_name),
        "dataset_label": str(args.dataset_label),
        "input_device": "vr",
        "num_episodes": int(args.num_episodes),
        "num_rows": int(len(rows_obs)),
        "obs_dim": int(rows_obs[0].shape[0]) if rows_obs else 0,
        "act_dim": int(rows_actions[0].shape[0]) if rows_actions else 0,
        "include_goal": bool(args.include_goal),
        "include_relative_cube_features": bool(args.include_relative_cube_features),
        "relative_only_obs": bool(args.relative_only_obs),
        "reward_type": str(args.reward_type),
        "dense_reward_scale": float(args.dense_reward_scale),
        "step_penalty": float(args.step_penalty),
        "teacher_target_mode": str(args.teacher_target_mode),
        "cube_success_tolerance": float(args.cube_success_tolerance),
        "physics_timestep": float(args.physics_timestep),
        "control_timestep": float(args.control_timestep),
        "created_at": float(time.time()),
    }
    saved = save_transition_dataset(
        path=dataset_path,
        metadata=metadata,
        observations=np.stack(rows_obs, axis=0) if rows_obs else np.zeros((0, 0), dtype=np.float32),
        actions=np.stack(rows_actions, axis=0) if rows_actions else np.zeros((0, 0), dtype=np.float32),
        next_observations=np.stack(rows_next_obs, axis=0) if rows_next_obs else np.zeros((0, 0), dtype=np.float32),
        rewards=np.asarray(rows_rewards, dtype=np.float32),
        dones=np.asarray(rows_dones, dtype=np.bool_),
        truncations=np.asarray(rows_trunc, dtype=np.bool_),
        teacher_intervened=np.ones((len(rows_obs),), dtype=np.bool_),
    )
    print(f"saved dataset to {saved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
