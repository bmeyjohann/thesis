#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ogbench_utils.env_wrappers_common import FixedResetSeedWrapper
from ogbench_utils.env_wrappers_manip import (
    CubeTeacherInfoAdapter,
    ManipDetailedRewardWrapper,
    ManipDisableRotationActionWrapper,
)
from ogbench_utils.intervention_wrappers import InterventionWrapper
from ogbench_utils.manip_topdown import extract_manip_topdown_state
from ogbench_utils.vr_mapping_web import VRMappingWebServer, create_vr_source
from ogbench_utils.vr_teleop import (
    DEFAULT_VR_CACHE_PATH,
    DEFAULT_VR_MAPPING_PATH,
    DEFAULT_VR_MAPPING_WEB_PORT,
    DEFAULT_VR_PORT,
    VRManipActionMapper,
    VRManipMappingConfig,
    VRStatusPanel,
    VRTeleopInterface,
    apply_vr_mapping_profile,
)


def _pump_pygame_events() -> None:
    try:
        if pygame.get_init():
            pygame.event.pump()
    except Exception:
        pass


class NativeViewer:
    """MuJoCo passive viewer wrapper for manip environments."""

    def __init__(self):
        self.enabled = False

    def maybe_launch(self, env: gym.Env) -> None:
        launch_fn = getattr(env.unwrapped, "launch_passive_viewer", None)
        if not callable(launch_fn):
            return
        try:
            launch_fn(show_left_ui=False, show_right_ui=False)
            self.enabled = True
            print("native viewer: launched", flush=True)
        except Exception as exc:
            self.enabled = False
            print(f"native viewer warning: launch failed: {exc}", flush=True)

    def sync(self, env: gym.Env) -> None:
        if not self.enabled:
            return
        sync_fn = getattr(env.unwrapped, "sync_passive_viewer", None)
        if not callable(sync_fn):
            return
        try:
            sync_fn()
        except Exception as exc:
            self.enabled = False
            print(f"native viewer warning: sync failed, disabling viewer: {exc}", flush=True)

    def close(self, env: gym.Env) -> None:
        close_fn = getattr(env.unwrapped, "close_passive_viewer", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _format_action(x: Any) -> str:
    if x is None:
        return "-"
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{float(v):+.3f}" for v in arr.tolist()) + "]"


@dataclass
class RuntimeState:
    episode: int = 0
    step: int = 0
    reset_count: int = 0
    lines: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "available": True,
            "episode": int(self.episode),
            "step": int(self.step),
            "reset_count": int(self.reset_count),
            "lines": list(self.lines),
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dedicated VR manipulation calibration tool.")
    p.add_argument("--env_name", type=str, default="cube-double-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=-1,
        help="-1 disables TimeLimit for calibration; 0 uses the env default; positive values set an explicit cap.",
    )
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--control_timestep", type=float, default=0.02)
    p.add_argument("--physics_timestep", type=float, default=0.002)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--mujoco_gl", type=str, default="auto")
    p.add_argument("--action_scale", type=float, default=1.0)
    p.add_argument("--hold_targets_on_zero_action", action="store_true", default=False)
    p.add_argument("--noop_action_threshold", type=float, default=1e-6)
    p.add_argument("--disable_rotation", action="store_true", default=False)
    p.add_argument("--binary_gripper_actions", action="store_true", default=False)
    p.add_argument("--binary_gripper_threshold", type=float, default=0.0)
    p.add_argument("--reward_type", type=str, default="sparse", choices=["none", "sparse", "dense", "combined"])
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--teacher_target_mode", type=str, default="sequential", choices=["fixed", "sequential"])
    p.add_argument("--cube_success_tolerance", type=float, default=0.04)
    p.add_argument("--human_threshold", type=float, default=-1e-6)
    p.add_argument("--human_hold_time", type=float, default=0.25)
    p.add_argument("--static_reset_seed", type=int, default=-1)
    p.add_argument("--show_status_panel", action="store_true", default=True)
    p.add_argument("--no_show_status_panel", dest="show_status_panel", action="store_false")
    p.add_argument("--show_topdown_view", action="store_true", default=True)
    p.add_argument("--no_show_topdown_view", dest="show_topdown_view", action="store_false")
    p.add_argument("--show_web", action="store_true", default=True)
    p.add_argument("--no_show_web", dest="show_web", action="store_false")
    p.add_argument("--web_host", type=str, default="127.0.0.1")
    p.add_argument("--web_port", type=int, default=DEFAULT_VR_MAPPING_WEB_PORT)
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
    p.add_argument("--vr_position_gain", type=float, default=25.0)
    p.add_argument("--vr_position_response_gain", type=float, default=1.5)
    p.add_argument("--vr_position_feedforward", type=float, default=0.5)
    p.add_argument("--vr_yaw_gain", type=float, default=2.5)
    p.add_argument("--vr_yaw_response_gain", type=float, default=1.5)
    p.add_argument("--vr_yaw_feedforward", type=float, default=0.5)
    p.add_argument("--vr_gripper_gain", type=float, default=5.0)
    p.add_argument("--vr_gripper_axis", type=str, default="trigger")
    p.add_argument("--vr_motion_control_mode", type=str, default="target_hold")
    p.add_argument("--vr_gripper_control_mode", type=str, default="absolute")
    p.add_argument("--vr_binary_gripper", action="store_true", default=False)
    p.add_argument("--vr_trigger_close_threshold", type=float, default=0.6)
    p.add_argument("--vr_trigger_open_threshold", type=float, default=0.2)
    p.add_argument("--vr_mirror_gripper_when_inactive", action="store_true", default=False)
    p.add_argument("--vr_invert_x", action="store_true", default=False)
    p.add_argument("--vr_invert_y", action="store_true", default=False)
    p.add_argument("--vr_invert_z", action="store_true", default=False)
    p.add_argument("--vr_invert_yaw", action="store_true", default=False)
    p.add_argument("--vr_invert_gripper", action="store_true", default=False)
    return p.parse_args()


def _create_env(args: argparse.Namespace, teleop: VRTeleopInterface) -> gym.Env:
    if not args.headless and args.render_mode == "human":
        os.environ["MUJOCO_GL"] = "glfw" if args.mujoco_gl == "auto" else args.mujoco_gl
    elif args.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    import ogbench  # noqa: F401

    render_mode = "rgb_array" if args.headless else (None if args.render_mode == "human" else args.render_mode)
    make_kwargs: dict[str, Any] = {"render_mode": render_mode}
    if float(args.physics_timestep) > 0.0:
        make_kwargs["physics_timestep"] = float(args.physics_timestep)
    if float(args.control_timestep) > 0.0:
        make_kwargs["control_timestep"] = float(args.control_timestep)
    make_kwargs["hold_targets_on_zero_action"] = bool(args.hold_targets_on_zero_action)
    make_kwargs["noop_action_threshold"] = float(args.noop_action_threshold)
    make_kwargs["disable_rotation"] = bool(args.disable_rotation)
    if int(args.max_episode_steps) == -1:
        make_kwargs["max_episode_steps"] = -1
    elif int(args.max_episode_steps) > 0:
        make_kwargs["max_episode_steps"] = int(args.max_episode_steps)
    env = gym.make(args.env_name, **make_kwargs)
    if int(args.static_reset_seed) >= 0:
        env = FixedResetSeedWrapper(env, reset_seed=int(args.static_reset_seed))
    env = ManipDetailedRewardWrapper(
        env,
        reward_type=str(args.reward_type),
        dense_reward_scale=float(args.dense_reward_scale),
        step_penalty=float(args.step_penalty),
    )
    if "cube" in str(args.env_name).lower():
        env = CubeTeacherInfoAdapter(
            env,
            target_mode=str(args.teacher_target_mode),
            success_tolerance=float(args.cube_success_tolerance),
        )
    env = InterventionWrapper(
        env,
        teleop_interface=teleop,
        mode="human",
        threshold=float(args.human_threshold),
        hold_time=float(args.human_hold_time),
        binary_gripper_actions=bool(args.binary_gripper_actions),
        binary_gripper_threshold=float(args.binary_gripper_threshold),
    )
    if bool(args.disable_rotation):
        env = ManipDisableRotationActionWrapper(env)
    return env


def _clip_action(action: np.ndarray, *, action_scale: float, binary_gripper_actions: bool, binary_gripper_threshold: float) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out *= float(action_scale)
    np.clip(out, -1.0, 1.0, out=out)
    if binary_gripper_actions and out.shape[-1] >= 4:
        gripper_idx = out.shape[-1] - 1
        out[gripper_idx] = 1.0 if float(out[gripper_idx]) >= float(binary_gripper_threshold) else -1.0
    return out


def main() -> int:
    args = _parse_args()
    mapping_path = Path(args.vr_mapping_path)
    runtime = RuntimeState()
    source = None
    web_server = None
    status_panel = None
    env = None
    viewer = NativeViewer()

    try:
        source = create_vr_source(
            vr_mode=args.vr_mode,
            vr_host=args.vr_host,
            vr_port=args.vr_port,
            cache_path=args.vr_cache_path,
            reconnect_seconds=args.vr_reconnect_seconds,
        )
        print(source.banner_text(), flush=True)

        mapping_config = VRManipMappingConfig(
            hand=args.vr_hand,
            require_gate=bool(args.vr_require_gate),
            gate_button=args.vr_gate_button,
            gripper_mirror_toggle_button=args.vr_gripper_mirror_toggle_button,
            rotation_source="global_yaw",
            motion_control_mode=str(args.vr_motion_control_mode),
            position_gain=float(args.vr_position_gain),
            position_response_gain=float(args.vr_position_response_gain),
            position_feedforward=float(args.vr_position_feedforward),
            yaw_gain=float(args.vr_yaw_gain),
            yaw_response_gain=float(args.vr_yaw_response_gain),
            yaw_feedforward=float(args.vr_yaw_feedforward),
            gripper_gain=float(args.vr_gripper_gain),
            trigger_axis=args.vr_gripper_axis,
            gripper_control_mode=str(args.vr_gripper_control_mode),
            binary_gripper=bool(args.vr_binary_gripper),
            trigger_close_threshold=float(args.vr_trigger_close_threshold),
            trigger_open_threshold=float(args.vr_trigger_open_threshold),
            mirror_gripper_when_inactive=bool(args.vr_mirror_gripper_when_inactive),
            invert_x=bool(args.vr_invert_x),
            invert_y=bool(args.vr_invert_y),
            invert_z=bool(args.vr_invert_z),
            invert_yaw=bool(args.vr_invert_yaw),
            invert_gripper=bool(args.vr_invert_gripper),
        )
        if bool(args.vr_use_saved_mapping):
            mapping_config, loaded = apply_vr_mapping_profile(mapping_config, mapping_path)
            if loaded:
                print(f"loaded vr mapping profile from {mapping_path}", flush=True)
        mapper_action_dim = 4 if bool(args.disable_rotation) else 5
        mapper = VRManipActionMapper(action_dim=mapper_action_dim, config=mapping_config)
        teleop = VRTeleopInterface(
            source,
            mapper,
            return_none_when_idle=True,
            idle_threshold=1e-6,
            mapping_path=mapping_path,
        )

        if bool(args.show_web):
            web_server = VRMappingWebServer(
                host=str(args.web_host),
                port=int(args.web_port),
                mapping_path=mapping_path,
                source=source,
                action_dim=mapper_action_dim,
                runtime_status_provider=runtime.payload,
            )
            web_server.start_background()
            print(f"vr mapping web ui listening at {web_server.url}", flush=True)
            print(f"mapping profile path: {mapping_path}", flush=True)

        if (not args.headless) and (bool(args.show_status_panel) or bool(args.show_topdown_view)):
            try:
                status_panel = VRStatusPanel(
                    title="VR Receiver Monitor",
                    action_dim=mapper_action_dim,
                    mapping_config=mapper.config,
                    mapping_path=mapping_path,
                    show_topdown=bool(args.show_topdown_view),
                )
            except Exception as exc:
                print(f"status panel unavailable: {exc}", flush=True)
                status_panel = None

        env = _create_env(args, teleop)
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"Expected Box action space, got {type(action_space)}")
        action_dim = int(np.prod(action_space.shape))
        zero_action = np.zeros((action_dim,), dtype=np.float32)
        if not args.headless and args.render_mode == "human":
            viewer.maybe_launch(env)

        print("=== VR Manip Config ===", flush=True)
        print(f"env={args.env_name}", flush=True)
        print(f"vr_transport={args.vr_mode}", flush=True)
        print(f"vr_mapping_path={mapping_path}", flush=True)
        print(f"web_ui={web_server.url if web_server is not None else '<disabled>'}", flush=True)
        print(
            f"timing physics_dt={float(args.physics_timestep):.4f}s control_dt={float(args.control_timestep):.4f}s "
            f"target_fps={float(args.fps):.1f}",
            flush=True,
        )
        print(
            f"human_intervention threshold={float(args.human_threshold):.3f} hold_time={float(args.human_hold_time):.3f}",
            flush=True,
        )
        print("controls: close the monitor window or press Q/ESC in it to quit, RIGHT/ENTER to force reset, UP/DOWN to change fps", flush=True)

        target_fps = max(1.0, float(args.fps))
        frame_dt = 1.0 / target_fps
        episode_idx = 0
        quit_requested = False
        last_summary = ""

        obs, info = env.reset(seed=int(args.seed) + episode_idx)
        teleop.update_robot_state(info if isinstance(info, dict) else None)
        runtime.reset_count += 1
        runtime.episode = episode_idx + 1
        runtime.step = 0
        while not quit_requested:
            t0 = time.perf_counter()
            _pump_pygame_events()
            snapshot = source.snapshot()
            force_reset = False
            if status_panel is not None:
                status_panel.set_snapshot(snapshot)
                status_panel.set_topdown_state(extract_manip_topdown_state(env, info if isinstance(info, dict) else None))
                _prev, next_requested, panel_quit, advance_requested, fps_delta = status_panel.poll()
                status_panel.draw()
                quit_requested = bool(panel_quit)
                force_reset = bool(next_requested or advance_requested)
                if fps_delta != 0.0:
                    target_fps = min(240.0, max(1.0, target_fps + float(fps_delta)))
                    frame_dt = 1.0 / target_fps

            action = _clip_action(
                zero_action,
                action_scale=float(args.action_scale),
                binary_gripper_actions=bool(args.binary_gripper_actions),
                binary_gripper_threshold=float(args.binary_gripper_threshold),
            )
            obs, reward, terminated, truncated, info = env.step(action)
            teleop.update_robot_state(info if isinstance(info, dict) else None)
            if not args.headless and args.render_mode == "human":
                viewer.sync(env)

            runtime.step += 1
            vr_diag = teleop.get_last_diag()
            intervened = bool(info.get("teacher_intervened", False)) if isinstance(info, dict) else False
            reason = str(info.get("teacher_reason", "-") or "-") if isinstance(info, dict) else "-"
            component = str(info.get("teacher_reason_component", "none") or "none") if isinstance(info, dict) else "none"
            student_action = info.get("student_action") if isinstance(info, dict) else None
            teacher_action = info.get("teacher_action") if isinstance(info, dict) else None
            applied_action = teacher_action if intervened else student_action
            reward_value = float(reward)
            success = int(bool(info.get("success", False))) if isinstance(info, dict) else 0
            target_block = int(info.get("privileged/target_block", -1)) if isinstance(info, dict) else -1
            runtime.lines = [
                f"intervened={int(intervened)} reason={reason} component={component} success={success} reward={reward_value:+.4f}",
                f"student_action={_format_action(student_action)}",
                f"applied_action={_format_action(applied_action)}",
                f"teacher_action={_format_action(teacher_action)} target_block={target_block}",
                (
                    f"vr connected={int(bool(vr_diag.get('connected', False)))} tracked={int(bool(vr_diag.get('tracked', False)))} "
                    f"gate={int(bool(vr_diag.get('gate_pressed', False)))} motion={int(bool(vr_diag.get('motion_active', False)))} "
                    f"trigger={float(vr_diag.get('trigger_value', 0.0)):.3f}"
                ),
                (
                    f"raw_dxyz=[{float(vr_diag.get('raw_delta_position', [0.0, 0.0, 0.0])[0]):+.4f}, "
                    f"{float(vr_diag.get('raw_delta_position', [0.0, 0.0, 0.0])[1]):+.4f}, "
                    f"{float(vr_diag.get('raw_delta_position', [0.0, 0.0, 0.0])[2]):+.4f}] "
                    f"drot={float(vr_diag.get('raw_delta_rotation', 0.0)):+.4f}"
                ),
                (
                    f"target_pose=[{float(vr_diag.get('target_robot_position', [0.0, 0.0, 0.0])[0]):+.3f}, "
                    f"{float(vr_diag.get('target_robot_position', [0.0, 0.0, 0.0])[1]):+.3f}, "
                    f"{float(vr_diag.get('target_robot_position', [0.0, 0.0, 0.0])[2]):+.3f}] "
                    f"target_rot={float(vr_diag.get('target_robot_rotation', 0.0)):+.3f} "
                    f"target_grip={float(vr_diag.get('target_robot_gripper', 0.0)):+.3f}"
                ),
                (
                    f"control_modes motion={vr_diag.get('motion_control_mode', '-')} "
                    f"gripper={vr_diag.get('gripper_control_mode', '-')} "
                    f"robot_state={int(bool(vr_diag.get('robot_state_available', False)))} "
                    f"current_grip={float(np.asarray(info.get('proprio/gripper_opening', [0.0]), dtype=np.float32).reshape(-1)[0]) if isinstance(info, dict) and 'proprio/gripper_opening' in info else 0.0:+.3f}"
                ),
                (
                    f"servo xyz_resp={float(vr_diag.get('position_response_gain', 0.0)):.2f} "
                    f"xyz_ff={float(vr_diag.get('position_feedforward', 0.0)):.2f} "
                    f"yaw_resp={float(vr_diag.get('yaw_response_gain', 0.0)):.2f} "
                    f"yaw_ff={float(vr_diag.get('yaw_feedforward', 0.0)):.2f}"
                ),
            ]

            if runtime.step % max(1, int(args.print_every)) == 0 or force_reset:
                summary = (
                    f"ep={runtime.episode:03d} step={runtime.step:04d} "
                    f"intervened={int(intervened)} reason={reason} component={component} "
                    f"applied={_format_action(applied_action)} "
                    f"vr[connected={int(bool(vr_diag.get('connected', False)))},tracked={int(bool(vr_diag.get('tracked', False)))},"
                    f"gate={int(bool(vr_diag.get('gate_pressed', False)))},motion={int(bool(vr_diag.get('motion_active', False)))},"
                    f"trigger={float(vr_diag.get('trigger_value', 0.0)):.3f}]"
                )
                if summary != last_summary:
                    print(summary, flush=True)
                    last_summary = summary

            should_reset = bool(force_reset or terminated or truncated or success)
            if should_reset and not quit_requested:
                if force_reset:
                    print("reset requested", flush=True)
                episode_idx += 1
                obs, info = env.reset(seed=int(args.seed) + episode_idx)
                teleop.reset()
                teleop.update_robot_state(info if isinstance(info, dict) else None)
                if status_panel is not None:
                    status_panel.set_topdown_state(extract_manip_topdown_state(env, info if isinstance(info, dict) else None))
                runtime.reset_count += 1
                runtime.episode = episode_idx + 1
                runtime.step = 0
                if not args.headless and args.render_mode == "human":
                    viewer.sync(env)

            elapsed = time.perf_counter() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    finally:
        if status_panel is not None:
            status_panel.close()
        if env is not None:
            viewer.close(env)
            env.close()
        if source is not None:
            source.close()
        if web_server is not None:
            web_server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
