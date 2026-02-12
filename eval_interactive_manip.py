#!/usr/bin/env python3
"""
Standalone interactive evaluator for OGBench manipulation tasks.

Focus:
- State observations only (fast debugging path).
- Random/keyboard controllers.
- Intervention diagnostics (teacher candidate availability, l2/angle deltas, reasons).

Example:
  python eval_interactive_manip.py \
    --env_name cube-double-v0 \
    --controller random \
    --intervention_mode agent \
    --teacher_type cube_plan \
    --tolerance_type l2 \
    --tolerance_value 0.02
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame

from ogbench_utils import build_ogbench_wrapper


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Convert quaternion (w, x, y, z) to yaw (z-axis Euler angle)."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


class CubeTeacherInfoAdapter(gym.Wrapper):
    """
    Inject target-block fields expected by cube teacher oracles.

    Some cube env variants expose block state in info but omit:
    - privileged/target_block
    - privileged/target_block_pos
    - privileged/target_block_yaw
    """

    def __init__(self, env: gym.Env, *, target_mode: str = "sequential", success_tolerance: float = 0.04):
        super().__init__(env)
        if target_mode not in {"fixed", "sequential"}:
            raise ValueError(f"Unknown target_mode={target_mode}")
        self.target_mode = target_mode
        self.success_tolerance = float(success_tolerance)

    def _cube_target_errors(self, out: dict, unwrapped) -> np.ndarray:
        num_cubes = int(getattr(unwrapped, "_num_cubes", 0))
        if num_cubes <= 0:
            return np.zeros(0, dtype=np.float32)
        errs = []
        for i in range(num_cubes):
            try:
                obj = np.asarray(out[f"privileged/block_{i}_pos"], dtype=np.float32)
            except Exception:
                obj = np.asarray(unwrapped._data.joint(f"object_joint_{i}").qpos[:3], dtype=np.float32)
            try:
                mocap_id = int(unwrapped._cube_target_mocap_ids[i])
                tar = np.asarray(unwrapped._data.mocap_pos[mocap_id], dtype=np.float32)
            except Exception:
                tar = obj
            errs.append(float(np.linalg.norm(obj - tar)))
        return np.asarray(errs, dtype=np.float32)

    def _select_target_block(self, out: dict, unwrapped, errs: np.ndarray) -> int:
        base_target = int(getattr(unwrapped, "_target_block", 0))
        if self.target_mode != "sequential" or errs.size == 0:
            return base_target
        unresolved = np.where(errs > self.success_tolerance)[0]
        if unresolved.size == 0:
            return base_target
        return int(unresolved[0])

    def _augment_info(self, info):
        if not isinstance(info, dict):
            return info
        out = dict(info)
        unwrapped = self.unwrapped
        errs = self._cube_target_errors(out, unwrapped)
        out["diag/cube_target_errors"] = errs
        out["diag/cubes_solved"] = int(np.sum(errs <= self.success_tolerance)) if errs.size else 0
        out["diag/cube_max_target_error"] = float(np.max(errs)) if errs.size else 0.0

        target_block = self._select_target_block(out, unwrapped, errs)
        out["privileged/target_block"] = int(target_block)
        out["diag/target_block_dynamic"] = int(target_block)

        try:
            target_idx = int(out["privileged/target_block"])
        except Exception:
            return out

        if target_idx < 0:
            return out

        try:
            mocap_pos = np.asarray(unwrapped._data.mocap_pos, dtype=np.float32)
            if target_idx < mocap_pos.shape[0] and "privileged/target_block_pos" not in out:
                out["privileged/target_block_pos"] = mocap_pos[target_idx].copy()
        except Exception:
            pass

        try:
            mocap_quat = np.asarray(unwrapped._data.mocap_quat, dtype=np.float32)
            if target_idx < mocap_quat.shape[0] and "privileged/target_block_yaw" not in out:
                yaw = _yaw_from_quat_wxyz(mocap_quat[target_idx])
                out["privileged/target_block_yaw"] = np.asarray([yaw], dtype=np.float32)
        except Exception:
            pass

        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment_info(info)


@dataclass
class StepDiagnostics:
    candidate: bool
    intervened: bool
    reason: Optional[str]
    delta_l2: float
    delta_angle_deg: float
    tolerance: float
    target_block: int
    cubes_solved: int
    cube_max_error: float


class RandomController:
    def __init__(self, action_space: gym.spaces.Box):
        self.action_space = action_space

    def action(self) -> np.ndarray:
        return np.asarray(self.action_space.sample(), dtype=np.float32)

    def close(self) -> None:
        return None


class IdleController:
    """Always output zero action."""

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def action(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        return None


class KeyboardController:
    """
    Keyboard mapping for 5D manip action spaces:
    - left/right arrows: x
    - up/down arrows: y
    - w/s: z
    - q/e: wrist yaw
    - a/d: gripper close/open
    """

    def __init__(self, action_dim: int, magnitude: float = 1.0):
        self.action_dim = int(action_dim)
        self.magnitude = float(magnitude)
        pygame.init()
        self._screen = pygame.display.set_mode((540, 140))
        pygame.display.set_caption("Manip Keyboard Controls")
        self._font = pygame.font.SysFont("Arial", 18)

    def _draw_help(self) -> None:
        self._screen.fill((20, 20, 20))
        lines = [
            "Controls: arrows=XY, W/S=Z, Q/E=Yaw, A/D=Gripper, SPACE=reset, N=skip, ESC/Q=quit",
            "Focus this control window for keyboard input. Env view is in native MuJoCo window.",
        ]
        y = 24
        for line in lines:
            surf = self._font.render(line, True, (230, 230, 230))
            self._screen.blit(surf, (10, y))
            y += 30
        pygame.display.flip()

    def action(self) -> tuple[np.ndarray, bool, bool, bool]:
        quit_requested = False
        reset_requested = False
        skip_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_requested = True
                if event.key == pygame.K_SPACE:
                    reset_requested = True
                if event.key == pygame.K_n:
                    skip_requested = True

        keys = pygame.key.get_pressed()
        a = np.zeros(self.action_dim, dtype=np.float32)
        mag = self.magnitude

        if self.action_dim >= 1:
            if keys[pygame.K_LEFT]:
                a[0] -= mag
            if keys[pygame.K_RIGHT]:
                a[0] += mag
        if self.action_dim >= 2:
            if keys[pygame.K_UP]:
                a[1] += mag
            if keys[pygame.K_DOWN]:
                a[1] -= mag
        if self.action_dim >= 3:
            if keys[pygame.K_w]:
                a[2] += mag
            if keys[pygame.K_s]:
                a[2] -= mag
        if self.action_dim >= 4:
            if keys[pygame.K_q]:
                a[3] -= mag
            if keys[pygame.K_e]:
                a[3] += mag
        if self.action_dim >= 5:
            if keys[pygame.K_a]:
                a[4] += mag
            if keys[pygame.K_d]:
                a[4] -= mag

        self._draw_help()
        return a, reset_requested, skip_requested, quit_requested

    def close(self) -> None:
        pygame.quit()


class NativeViewer:
    """MuJoCo passive viewer wrapper for manip environments."""

    def __init__(self):
        self.enabled = False

    def maybe_launch(self, env: gym.Env) -> None:
        base = env.unwrapped
        launch_fn = getattr(base, "launch_passive_viewer", None)
        if not callable(launch_fn):
            return
        try:
            launch_fn(show_left_ui=False, show_right_ui=False)
            self.enabled = True
            print("native viewer: launched")
        except Exception as exc:
            self.enabled = False
            print(f"native viewer warning: launch failed: {exc}")

    def sync(self, env: gym.Env) -> None:
        if not self.enabled:
            return
        base = env.unwrapped
        sync_fn = getattr(base, "sync_passive_viewer", None)
        if not callable(sync_fn):
            return
        try:
            sync_fn()
        except Exception as exc:
            self.enabled = False
            print(f"native viewer warning: sync failed, disabling viewer: {exc}")

    def close(self, env: gym.Env) -> None:
        base = env.unwrapped
        close_fn = getattr(base, "close_passive_viewer", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def clip_action(action: np.ndarray, *, clip_l2: bool = True) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out = np.clip(out, -1.0, 1.0)
    if clip_l2:
        norm = float(np.linalg.norm(out))
        if norm > 1.0 and norm > 1e-8:
            out = out / norm
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone manipulation intervention evaluator")
    p.add_argument("--env_name", type=str, default="cube-double-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=1, help="0 => run forever")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--controller", type=str, default="random", choices=["random", "keyboard", "human", "idle"])
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--mujoco_gl", type=str, default="auto", choices=["auto", "glfw", "egl"])
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--headless", action="store_true", default=False)

    p.add_argument("--obs_mode", type=str, default="state", choices=["state"])
    p.add_argument("--include_goal", action="store_true", default=True)
    p.add_argument("--include_distance", action="store_true", default=False)
    p.add_argument("--include_direction", action="store_true", default=False)
    p.add_argument("--include_velocity", action="store_true", default=False)

    p.add_argument("--reward_type", type=str, default="sparse", choices=["sparse", "dense", "combined", "none"])
    p.add_argument("--dense_reward_scale", type=float, default=0.01)
    p.add_argument("--step_penalty", type=float, default=0.0)

    p.add_argument("--intervention_mode", type=str, default="none", choices=["none", "agent"])
    p.add_argument("--teacher_type", type=str, default="cube_plan", choices=["cube_plan", "cube_markov"])
    p.add_argument("--tolerance_type", type=str, default="l2", choices=["l2", "angle"])
    p.add_argument("--tolerance_value", type=float, default=0.02)
    p.add_argument("--intervention_enable_after_steps", type=int, default=0)
    p.add_argument("--hard_block_lethal", action="store_true", default=False)

    p.add_argument("--action_scale", type=float, default=1.0)
    p.add_argument("--keyboard_scale", type=float, default=0.5)
    p.add_argument("--no_clip_action_l2", action="store_true", default=False)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--print_interventions", action="store_true", default=True)
    p.add_argument("--print_info_keys_once", action="store_true", default=False)
    p.add_argument("--teacher_target_mode", type=str, default="sequential", choices=["fixed", "sequential"])
    p.add_argument("--cube_success_tolerance", type=float, default=0.04)
    return p.parse_args()


def _extract_diag(info: dict, default_tol: float) -> StepDiagnostics:
    return StepDiagnostics(
        candidate=bool(info.get("teacher_candidate_available", False)),
        intervened=bool(info.get("teacher_intervened", False)),
        reason=info.get("teacher_reason"),
        delta_l2=float(info.get("teacher_delta_l2", 0.0)),
        delta_angle_deg=float(info.get("teacher_delta_angle_deg", 0.0)),
        tolerance=float(info.get("teacher_tolerance_value", default_tol)),
        target_block=int(info.get("privileged/target_block", 0)),
        cubes_solved=int(info.get("diag/cubes_solved", 0)),
        cube_max_error=float(info.get("diag/cube_max_target_error", 0.0)),
    )


def create_env(args: argparse.Namespace) -> gym.Env:
    if not args.headless and args.render_mode == "human":
        if args.mujoco_gl == "auto":
            os.environ["MUJOCO_GL"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = args.mujoco_gl
    elif args.mujoco_gl != "auto":
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    # Import after MUJOCO_GL is configured so MuJoCo backend selection is respected.
    import ogbench  # noqa: F401

    # Manip env ignores render_mode and always returns rgb arrays from render().
    # Use None for interactive mode and native passive viewer instead.
    render_mode = "rgb_array" if args.headless else (None if args.render_mode == "human" else args.render_mode)
    env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode=render_mode)
    env = CubeTeacherInfoAdapter(
        env,
        target_mode=args.teacher_target_mode,
        success_tolerance=args.cube_success_tolerance,
    )
    env = build_ogbench_wrapper(
        obs_mode=args.obs_mode,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        intervention_mode=args.intervention_mode,
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
    )(env)
    return env


def run(args: argparse.Namespace) -> None:
    if not args.headless and args.render_mode != "human":
        print("For interactive manipulation debugging, forcing native render_mode=human (avoids flicker).")
        args.render_mode = "human"

    env = create_env(args)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"Expected Box action space, got {type(action_space)}")

    action_dim = int(np.prod(action_space.shape))
    if args.controller == "random":
        controller = RandomController(action_space)
    elif args.controller == "idle":
        controller = IdleController(action_dim)
    else:
        # "human" is an alias for keyboard teleop.
        controller = KeyboardController(action_dim, magnitude=float(args.keyboard_scale))
    native_viewer = NativeViewer()

    print("=== Manip Eval ===")
    print(f"env={args.env_name}")
    print(f"controller={args.controller}")
    print(f"teacher={args.teacher_type}, mode={args.intervention_mode}")
    print(f"tolerance={args.tolerance_type}:{args.tolerance_value}")
    print(f"target_mode={args.teacher_target_mode}, cube_success_tol={args.cube_success_tolerance}")
    print(f"action_dim={int(np.prod(action_space.shape))}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    if args.intervention_mode != "none":
        print("note: teacher interventions are enabled; agent movement may occur without keyboard input.")
    print("")

    episode_idx = 0
    quit_requested = False
    frame_dt = 1.0 / max(1, args.fps)
    total_candidate = 0
    total_intervened = 0
    total_success_episodes = 0

    try:
        while (args.num_episodes == 0 or episode_idx < args.num_episodes) and not quit_requested:
            obs, info = env.reset(seed=args.seed + episode_idx)
            if not args.headless and args.render_mode == "human":
                if not native_viewer.enabled:
                    native_viewer.maybe_launch(env)
                native_viewer.sync(env)
            episode_idx += 1
            ep_reward = 0.0
            ep_len = 0
            ep_candidate = 0
            ep_intervened = 0
            ep_last_success = False
            done = False
            print(f"\n[Episode {episode_idx}]")

            while not done and not quit_requested:
                t0 = time.perf_counter()
                if args.controller in ("random", "idle"):
                    action = controller.action()
                    reset_requested = False
                    skip_requested = False
                else:
                    action, reset_requested, skip_requested, quit_requested = controller.action()

                if reset_requested:
                    print("manual reset requested")
                    break
                if skip_requested:
                    print("skip requested: moving to next episode")
                    done = True
                    continue

                action = clip_action(
                    action * float(args.action_scale),
                    clip_l2=(not args.no_clip_action_l2),
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                if not args.headless and args.render_mode == "human":
                    native_viewer.sync(env)

                ep_reward += float(reward)
                ep_len += 1

                diag = _extract_diag(info if isinstance(info, dict) else {}, args.tolerance_value)
                success_val = False
                if isinstance(info, dict):
                    raw_success = info.get("success", False)
                    if isinstance(raw_success, (np.ndarray, list, tuple)):
                        raw_arr = np.asarray(raw_success).reshape(-1)
                        success_val = bool(raw_arr[0]) if raw_arr.size else False
                    else:
                        success_val = bool(raw_success)
                ep_last_success = success_val
                if diag.candidate:
                    ep_candidate += 1
                if diag.intervened:
                    ep_intervened += 1

                if args.print_interventions and diag.intervened:
                    print(
                        f"step={ep_len:04d} intervention reason={diag.reason} "
                        f"r={float(reward):+.3f} success={int(success_val)} "
                        f"target={diag.target_block} solved={diag.cubes_solved} max_err={diag.cube_max_error:.4f} "
                        f"l2={diag.delta_l2:.4f} angle={diag.delta_angle_deg:.2f} tol={diag.tolerance:.4f}"
                    )
                elif ep_len % max(1, args.print_every) == 0:
                    print(
                        f"step={ep_len:04d} r={float(reward):+.3f} "
                        f"success={int(success_val)} "
                        f"target={diag.target_block} solved={diag.cubes_solved} max_err={diag.cube_max_error:.4f} "
                        f"cand={int(diag.candidate)} intv={int(diag.intervened)} "
                        f"l2={diag.delta_l2:.4f} angle={diag.delta_angle_deg:.2f} tol={diag.tolerance:.4f}"
                    )
                if args.print_info_keys_once and ep_len == 1 and isinstance(info, dict):
                    print(f"info_keys={sorted(info.keys())}")

                elapsed = time.perf_counter() - t0
                if frame_dt > elapsed:
                    time.sleep(frame_dt - elapsed)

            total_candidate += ep_candidate
            total_intervened += ep_intervened
            total_success_episodes += int(ep_last_success)
            print(
                f"episode_reward={ep_reward:.3f} steps={ep_len} "
                f"success={int(ep_last_success)} "
                f"candidate_steps={ep_candidate} intervened_steps={ep_intervened}"
            )
            if isinstance(info, dict) and "teacher_num_interventions" in info:
                print(
                    "teacher_summary "
                    f"num_interventions={int(info.get('teacher_num_interventions', 0))} "
                    f"intervention_steps={int(info.get('teacher_intervention_steps', 0))} "
                    f"fraction={float(info.get('teacher_fraction_steps', 0.0)):.3f}"
                )

        print(
            f"\nDone. episodes={episode_idx} total_candidate_steps={total_candidate} "
            f"total_intervened_steps={total_intervened} "
            f"success_episodes={total_success_episodes}/{episode_idx if episode_idx > 0 else 1}"
        )
    finally:
        controller.close()
        native_viewer.close(env)
        env.close()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
